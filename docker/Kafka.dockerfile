FROM confluentinc/cp-kafka:7.5.0

# Expose Kafka ports
# 9092: External client connections
# 29092: Internal broker communication
# 9999: JMX monitoring port
EXPOSE 9092 29092 9999
