import random
import json
import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_SERVICE_KEY:
    log.error("SUPABASE_SERVICE_ROLE_KEY not found in environment")
    raise ValueError("Missing SUPABASE_SERVICE_ROLE_KEY - required for admin operations")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

DAYS = 14

# hotel room configurations
ROOMS = {
    "Presidential Suite": {'amenities': ['ocean_view', 'balcony', 'jacuzzi', 'butler_service', 'premium_minibar'], 'total': 1, 'image_url': "", "base_price": 450, 'name': 'Presidential Suite', 'refundable': True, 'max_occupancy': 4},
    "Executive Suite": {'amenities': ['city_view', 'balcony', 'workspace', 'lounge_access'], 'total': 2, 'image_url': "", "base_price": 280, 'name': 'Executive Suite', 'refundable': True, 'max_occupancy': 3},
    "Junior Suite": {'amenities': ['garden_view', 'mini_fridge', 'coffee_maker'], 'total': 5, 'image_url': "", "base_price": 180, 'name': 'Junior Suite', 'refundable': True, 'max_occupancy': 2},
    "Deluxe Room": {'amenities': ['city_view', 'work_desk', 'coffee_maker'], 'total': 8, 'image_url': "", "base_price": 140, 'name': 'Deluxe Room', 'refundable': False, 'max_occupancy': 2},
    "Superior Room": {'amenities': ['wifi', 'tv', 'safe'], 'total': 12, 'image_url': "", "base_price": 110, 'name': 'Superior Room', 'refundable': False, 'max_occupancy': 2},
    "Standard Room": {'amenities': ['wifi', 'tv'], 'total': 20, 'image_url': "", "base_price": 85, 'name': 'Standard Room', 'refundable': False, 'max_occupancy': 2},
}

# flight configurations
FLIGHTS = {
    "JFK-LAX-Economy": {'departure': {'time': '08:00', 'airport': 'JFK'}, 'arrival': {'time': '11:30', 'airport': 'LAX'}, 'duration': '5h 30m', 'stops': 0, 'cabin_class': 'economy', 'fare_rule': 'standard', 'refundable': False, 'total': 180, 'base_price': 250},
    "JFK-LAX-Business": {'departure': {'time': '08:00', 'airport': 'JFK'}, 'arrival': {'time': '11:30', 'airport': 'LAX'}, 'duration': '5h 30m', 'stops': 0, 'cabin_class': 'business', 'fare_rule': 'flexible', 'refundable': True, 'total': 30, 'base_price': 850},
    "ORD-MIA-Economy": {'departure': {'time': '14:15', 'airport': 'ORD'}, 'arrival': {'time': '18:45', 'airport': 'MIA'}, 'duration': '3h 30m', 'stops': 0, 'cabin_class': 'economy', 'fare_rule': 'basic', 'refundable': False, 'total': 200, 'base_price': 180},
    "SFO-SEA-Premium": {'departure': {'time': '06:30', 'airport': 'SFO'}, 'arrival': {'time': '08:45', 'airport': 'SEA'}, 'duration': '2h 15m', 'stops': 0, 'cabin_class': 'premium', 'fare_rule': 'standard', 'refundable': False, 'total': 60, 'base_price': 420},
    "ATL-DFW-First": {'departure': {'time': '16:00', 'airport': 'ATL'}, 'arrival': {'time': '17:30', 'airport': 'DFW'}, 'duration': '2h 30m', 'stops': 0, 'cabin_class': 'first', 'fare_rule': 'flexible', 'refundable': True, 'total': 12, 'base_price': 1600},
    "LAX-SFO-Economy": {'departure': {'time': '10:00', 'airport': 'LAX'}, 'arrival': {'time': '11:30', 'airport': 'SFO'}, 'duration': '1h 30m', 'stops': 0, 'cabin_class': 'economy', 'fare_rule': 'standard', 'refundable': False, 'total': 150, 'base_price': 120},
    "MIA-ATL-Premium": {'departure': {'time': '19:00', 'airport': 'MIA'}, 'arrival': {'time': '20:45', 'airport': 'ATL'}, 'duration': '1h 45m', 'stops': 0, 'cabin_class': 'premium', 'fare_rule': 'standard', 'refundable': True, 'total': 50, 'base_price': 380},
    "DFW-ORD-Economy": {'departure': {'time': '07:30', 'airport': 'DFW'}, 'arrival': {'time': '10:15', 'airport': 'ORD'}, 'duration': '2h 45m', 'stops': 0, 'cabin_class': 'economy', 'fare_rule': 'basic', 'refundable': False, 'total': 190, 'base_price': 160},
    "SEA-LAX-Business": {'departure': {'time': '13:00', 'airport': 'SEA'}, 'arrival': {'time': '15:30', 'airport': 'LAX'}, 'duration': '2h 30m', 'stops': 0, 'cabin_class': 'business', 'fare_rule': 'flexible', 'refundable': True, 'total': 40, 'base_price': 720},
    "LAX-JFK-First": {'departure': {'time': '18:00', 'airport': 'LAX'}, 'arrival': {'time': '02:15', 'airport': 'JFK'}, 'duration': '5h 15m', 'stops': 0, 'cabin_class': 'first', 'fare_rule': 'flexible', 'refundable': True, 'total': 16, 'base_price': 1850},
}

def gen_hotel_products():
    """generate hotel room products for next DAYS days"""
    data = []
    for day in range(DAYS):
        for room_type, rdata in ROOMS.items():
            data.append({
                'room_type': room_type,
                'date_index': day + 1,
                'metadata': rdata,
                'availability': random.randint(0, rdata['total'])
            })
    return data

def gen_airline_products():
    """generate flight products for next DAYS days"""
    data = []
    for day in range(DAYS):
        for flight_type, fdata in FLIGHTS.items():
            data.append({
                'flight_type': flight_type,
                'date_index': day + 1,
                'metadata': fdata,
                'availability': random.randint(0, fdata['total'])
            })
    return data

def clear_table(table_name: str):
    """clear all records from a table"""
    try:
        resp = supabase.table(table_name).select('id').execute()
        if resp.data:
            ids = [row['id'] for row in resp.data]
            chunk_size = 100
            for i in tqdm(range(0, len(ids), chunk_size), desc=f"Clearing {table_name}", unit="chunk"):
                chunk = ids[i:i+chunk_size]
                supabase.table(table_name).delete().in_('id', chunk).execute()
            log.info(f"Deleted {len(ids)} records from {table_name}")
        else:
            log.info(f"{table_name} already empty")
    except Exception as e:
        log.error(f"Failed to clear {table_name}: {e}")
        raise

def seed_table(table_name: str, data: list[dict]):
    """insert records into a table"""
    try:
        chunk_size = 100
        total = len(data)
        for i in tqdm(range(0, total, chunk_size), desc=f"Seeding {table_name}", unit="chunk"):
            chunk = data[i:i+chunk_size]
            supabase.table(table_name).insert(chunk).execute()
        log.info(f"Inserted {total} records into {table_name}")
    except Exception as e:
        log.error(f"Failed to seed {table_name}: {e}")
        raise

def main():

    log.info("Generating hotel products...")
    hotel_products = gen_hotel_products()
    log.info(f"Generated {len(hotel_products)} hotel products")

    log.info("Generating airline products...")
    airline_products = gen_airline_products()
    log.info(f"Generated {len(airline_products)} airline products\n")

    log.info("Clearing existing products...")
    clear_table('hotel_products')
    clear_table('airline_products')

    log.info("Seeding products...")
    seed_table('hotel_products', hotel_products)
    seed_table('airline_products', airline_products)


if __name__ == "__main__":
    main()
