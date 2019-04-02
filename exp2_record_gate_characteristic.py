import smu
import numpy as np

s = smu.smu()

v = np.linspace(0,5,1000)

f = open("data/exp2_nmos_parallel_vdd.csv",'w')
f.write("Vg, I\n")

s.set_voltage(1,0)
s.autorange(1)
s.set_voltage(2,0)
s.autorange(2)
s.set_irange(2,0)

for val in v:
    s.set_voltage(1,val)
    s.autorange(1)
    s.set_voltage(2,5)
    #s.autorange(2)
    f.write('{!s},{!s}\n'.format(val,s.get_current(2)))

s.set_current(1,0)
f.close()
