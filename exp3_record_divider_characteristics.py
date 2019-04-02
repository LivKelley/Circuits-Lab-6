import smu
import numpy as np

s = smu.smu()

v = np.linspace(0,0.02,1000)

f = open("data/exp3_two_way.csv",'w')
f.write("Iin, I2\n")

s.set_current(1,0)
s.autorange(1)
s.set_voltage(2,0)
s.autorange(2)
s.set_irange(2,0)

for val in v:
    s.set_current(1,val)
    s.autorange(1)
    s.set_voltage(2,0)
    #s.autorange(2)
    f.write('{!s},{!s}\n'.format(val,s.get_current(2)))

s.set_current(1,0)
f.close()
