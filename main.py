import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from sympy.physics.optics import *
from sympy import Symbol, Matrix
from cavity import Cavity
import sys

inf_meter = 10000000.0

khz = 1e3
mhz = 1e6
ghz = 1e9
thz = 1e12

c = 299792458.0

ms = 1 / khz
us = 1 / mhz
ns = 1 / ghz

m = 1
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

ppm = 1e-6

# c1 = Cavity(roc1=-10*mm, roc2=-10*mm, pos1=0, d=10.001*mm, R1=0.98, R2=0.98, lamb=1042*nm, I0=1)
# c1Spectrum = c1.get_response(wCenter=1042 * nm, fWidth=30 * ghz, num=10000)
# plt.plot((c1Spectrum[0]-c/c1.lamb)/ghz,
#          c1Spectrum[1])
#
# c2 = Cavity(roc1=-inf_meter, roc2=-10.0*mm, pos1=0, d=5.0*mm, R1=0.9, R2=0.9, lamb=1042*nm, I0=1)
# c2Response = c2.get_response(wCenter=1042 * nm, fWidth=50 * ghz, num=10000)
# # plt.plot((c2Spectrum[0]-c/c2.lamb)/ghz,
# #          c2Spectrum[1])
# plt.plot((c2Response[0]-c/c2.lamb)/ghz, c2Response[2], label="Reflectance")
# plt.plot((c2Response[0]-c/c2.lamb)/ghz, c2Response[3], label="Transmission")
#
# plt.xlabel("Frequency, zero at 1042 nm (GHz)")
# plt.ylabel("I/I0")
#
# plt.legend()
# plt.tight_layout()
# plt.show()


L = 10*ppm
T = 90*ppm
R = 1-L-T
c3 = Cavity(roc1=-inf_meter, roc2=-10*mm, pos1=0, d=5.0*mm, R1=R, R2=R, L1=L, L2=L, lamb=1042*nm, I0=1)
#c3.report()

p = BeamParameter(wavelen=1042*nm, z=0, w=c3.W0) # z=0 -> q at the waist
print("q(z) = ", p.q.n(), ", w(z) = ", p.w.n()/um, ", w0 = ", p.w_0.n()/um)
p = FreeSpace(10*cm)*p # Apply a ray transfer matrix to the complex beam parameter p
print("q(z) = ", p.q.n(), ", w(z) = ", p.w.n()/um, ", w0 = ", p.w_0.n()/um)
p = ThinLens(10*cm)*p # Focal length 10 cm
p = FreeSpace(5*cm)*p
print("q(z) = ", p.q.n(), ", w(z) = ", p.w.n()/um, ", w0 = ", p.w_0.n()/um) # 5 cm after lens
p = FreeSpace(5*cm)*p
print("q(z) = ", p.q.n(), ", w(z) = ", p.w.n()/um, ", w0 = ", p.w_0.n()/um)

class BeamPlot:
    def __init__(self):
        pg.setConfigOptions(antialias=False)
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Beam plotter')
        self.win.setWindowTitle('Beam plotter')
        # self.win.setGeometry(5, 115, 1910, 1070)

        bf_xlabels = [(0, '0'), (0.5, '0.5'), (1, '1')]
        bf_xaxis = pg.AxisItem(orientation='bottom')
        bf_xaxis.setTicks([bf_xlabels])

        bf_ylabels = [(0, '0'), (128, '128'), (255, '255')]
        bf_yaxis = pg.AxisItem(orientation='left')
        bf_yaxis.setTicks([bf_ylabels])

        self.beam = self.win.addPlot(
            title='Beam', axisItems={'bottom': bf_xaxis, 'left': bf_yaxis}
        )
        """
        d1 = 25 * cm
        f1 = 25 * cm
        d2 = 50 * cm

        beam = BeamParameter(1042 * nm, 0, w=c3.W0)

        z_c = np.linspace(0, c3.d, 1000)


        z1 = np.linspace(0, d1, 1000)
        print("q(z) = ", beam.q.n()/cm, ", w(z) = ", beam.w.n() / um)

        w = float(beam.w_0.n()) * np.sqrt(1 + ((z1-z1[0]) / float(beam.z_r.n())) ** 2)
        self.beam.plot(z1, w)
        self.beam.plot(z1, -w)

        beam = ThinLens(f1)*FreeSpace(d1)*beam
        print("q(z) = ", beam.q.n()/cm, ", w(z) = ", beam.w.n() / um)

        z2 = np.linspace(d1, d1+d2, 1000)
        w = float(beam.w_0.n()) * np.sqrt(1 + ((z2-z2[0]+float(beam.z.n())) / float(beam.z_r.n())) ** 2)
        # Convert z2 into the position with respect to the waist
        self.beam.plot(z2, w)
        self.beam.plot(z2, -w)
        
        z3 = np.linspace(d1+d2, d1+d2+d2, 1000)
        beam = ThinLens(f1)*FreeSpace(d2)*beam
        print("q(z) = ", beam.q.n()/cm, ", w(z) = ", beam.w.n() / um)
                
        w = float(beam.w_0.n()) * np.sqrt(1 + ((z3-z3[0]+float(beam.z.n())) / float(beam.z_r.n())) ** 2)
        self.beam.plot(z3, w)
        self.beam.plot(z3, -w)
        
        self.beam.showGrid(x=True, y=True)
        """

    @staticmethod
    def start():
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

def EquiLens(R1,R2,d):
    
    return 0

if __name__ == '__main__':
    app = BeamPlot()
    app.start()