echo "Starting the Docker container...enter 'exit' to close when done."
# shifterimg pull kpkaur28/neuron:v3 - run this before running bash script
# shifter --image=kpkaur28/neuron:v3 /bin/bash
shifterimg pull adammwea/netpyneshifter:v6 #- run this before running bash script
shifter --image=adammwea/netpyneshifter:v6 /bin/bash
#cat /etc/lsb-release
#kpkaur28/neuron:v3