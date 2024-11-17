import sys
sys.path.append('submodules/netpyne')
import netpyne

def load_clean_sim_object(data_file_path):
        try: netpyne.sim.clearAll() #clear all sim data
        except: pass
        netpyne.sim.loadAll(data_file_path)