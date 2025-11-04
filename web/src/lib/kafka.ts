import { Kafka, Producer } from 'kafkajs';

let producer: Producer | null = null;

const kafka = new Kafka({
    clientId: 'phantom-web',
    brokers: [`${process.env.KAFKA_HOST || 'localhost'}:${process.env.KAFKA_PORT || '9092'}`],
});

export const getProducer = async (): Promise<Producer> => {
    if (!producer) {
        producer = kafka.producer();
        await producer.connect();
    }
    return producer;
};

export const sendInteractionEvent = async (ev: {
    sessionId: string;
    eventType: string;
    targetEl?: string;
    targetUrl?: string;
    metadata?: Record<string, any>;
    ts: number;
}) => {
    const p = await getProducer();
    // add to the metadata
    await p.send({
        topic: 'user-interactions',
        messages: [{
            key: ev.sessionId,
            value: JSON.stringify(ev),
        }],
    });
};

export const disconnect = async () => {
    if (producer) {
        await producer.disconnect();
        producer = null;
    }
};
