import { Kafka, Producer } from 'kafkajs';
import type { EventBase } from './events';

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

export const sendEvent = async (ev: EventBase) => {
  const p = await getProducer();
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
