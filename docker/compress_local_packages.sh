# local packages to compress:
# 1. /home/adamm/workspace/MEA_Analysis
# 2. //home/adamm/workspace/netpyne
# 3. /home/adamm/workspace/RBS_network_models

# compressing /home/adamm/workspace/MEA_Analysis
cd /home/adamm/workspace/MEA_Analysis
tar -czf /home/adamm/workspace/MEA_Analysis.tar.gz *
mv /home/adamm/workspace/MEA_Analysis.tar.gz /home/adamm/workspace/RBS_network_models/docker

# compressing //home/adamm/workspace/netpyne
cd /home/adamm/workspace/netpyne
tar -czf /home/adamm/workspace/netpyne.tar.gz *
mv /home/adamm/workspace/netpyne.tar.gz /home/adamm/workspace/RBS_network_models/docker

# cd /home/adamm/workspace/axon_reconstructor
cd /home/adamm/workspace/axon_reconstructor
tar -czf /home/adamm/workspace/axon_reconstructor.tar.gz *
mv /home/adamm/workspace/axon_reconstructor.tar.gz /home/adamm/workspace/RBS_network_models/docker

# compressing /home/adamm/workspace/RBS_network_models
cd /home/adamm/workspace/RBS_network_models
rm -rf home/adamm/workspace/docker/RBS_network_models.tar.gz
tar -czf /home/adamm/workspace/RBS_network_models.tar.gz *
mv /home/adamm/workspace/RBS_network_models.tar.gz /home/adamm/workspace/RBS_network_models/docker

