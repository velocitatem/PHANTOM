from supabase import create_client, Client
import os
import random
import asyncio
import json
from dotenv import load_dotenv

from experiments.agents.agent import get_agent, AgentTypes
from lib.kafka_client import get_interactions

load_dotenv()

RESULTS="/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"

client = create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
    os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
)
def pick_random_task():
    mode = 'hotel'
    tasks = client.table("tasks").select("*").execute().data
    if mode == 'hotel':
        # drop all that have 'flight' in the description
        tasks = [task for task in tasks if 'flight' not in task['task_description'].lower()]
    return random.choice(tasks) if tasks else None

def clear_kafka_data():
    """Delete and recreate Kafka topics to clear all data"""
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import UnknownTopicOrPartitionError
    import time

    kafka_host = os.getenv('KAFKA_HOST', 'localhost')
    kafka_port = os.getenv('KAFKA_PORT', '9092')
    broker = f'{kafka_host}:{kafka_port}'

    admin = KafkaAdminClient(bootstrap_servers=broker)
    topics = ['user-interactions', 'price-logs']

    try:
        admin.delete_topics(topics, timeout_ms=5000)
        print(f"Deleted topics: {topics}")
        time.sleep(2)
    except UnknownTopicOrPartitionError:
        print("Topics don't exist, skipping delete")
    except Exception as e:
        print(f"Error deleting topics: {e}")

    new_topics = [
        NewTopic(name='user-interactions', num_partitions=3, replication_factor=1),
        NewTopic(name='price-logs', num_partitions=3, replication_factor=1)
    ]

    try:
        admin.create_topics(new_topics=new_topics, validate_only=False)
        print(f"Recreated topics: {topics}")
    except Exception as e:
        print(f"Error creating topics: {e}")
    finally:
        admin.close()

def create_new_experiment(task_id):
    import uuid
    subject_name = f"agent_{str(uuid.uuid4())[:8]}"
    experiment = {
        "subject_name": subject_name,
        "xp_human_only": False,
        "xp_market_mode": "hotel",
        "xp_task_id": task_id,
    }
    response = client.table("experiments").insert(experiment).execute()
    return response.data[0] if response.data else None

if __name__ == "__main__":
    clear_kafka_data()

    task = pick_random_task()
    if not task:
        print("No tasks available")
        exit(1)

    experiment = create_new_experiment(task['id'])
    exp_id = experiment['id']
    exp_dir = f"{RESULTS}{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)

    # construct experiment URL with uuid param
    base_url = os.getenv('NEXT_PUBLIC_API_BASE', 'http://localhost:3000')
    agent_url = f"{base_url}/start-task?uuid={exp_id}"

    print(f"Created experiment {exp_id} for task {task['id']}")
    print(f"Agent will interact with: {agent_url}")

    # instantiate and run agent
    agent = get_agent(
        AgentTypes.GENERIC_BROWSER_USE_AGENT,
        goal=task['task_description'],
        url=agent_url,
        timeout=300,
        headless=True
    )

    result = asyncio.run(agent.act())
    print(f"Agent result: {result}")

    # export interaction and price data from kafka
    interactions = get_interactions(topic='user-interactions', timeout_ms=3000)
    prices = get_interactions(topic='price-logs', timeout_ms=3000)

    with open(f"{exp_dir}/int.json", 'w') as f:
        json.dump(interactions, f, indent=2)

    with open(f"{exp_dir}/price.json", 'w') as f:
        json.dump(prices, f, indent=2)

    print(f"Experiment {exp_id} completed.")
    print(f"Exported {len(interactions)} interactions and {len(prices)} price logs to {exp_dir}")
