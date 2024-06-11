#!/bin/bash

# Start munge and slurm services
/etc/init.d/munge start
/etc/init.d/slurmctld start
/etc/init.d/slurmd start

# Keep the container running
tail -f /dev/null