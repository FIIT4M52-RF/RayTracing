import PIL
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from random import randint


Cw = 200
Ch = 200
Vw = 1
Vh = 1
projection_plane_d = 1
O = np.array([0, 0, 0])
BACKGROUND_COLOR = np.array([0, 0, 0])
recursion_depth = 3


class Sphere:
  def __init__(self, center=np.array([0, 0, 0]), radius=1.0, color=np.array(BACKGROUND_COLOR), specular = -1, reflective = 0):
    self.__center = center
    self.__radius = radius
    self.__color=np.array(color)
    self.__specular = specular
    self.__reflective = reflective

  def GetRadius(self):
    return self.__radius
  def GetCenter(self):
    return self.__center
  def GetColor(self):
    return self.__color
  def GetSpecular(self):
    return self.__specular
  def GetReflective(self):
    return self.__reflective

class Light:
  def __init__(self,source_type="point",position=np.array([0.5,0.5,0.5]),intensity=0.5):
    self.__source_type=source_type
    self.__position= np.array(position)
    self.__intensity = intensity
      
  def GetIntensity(self):
    return self.__intensity
  def GetSourseType(self):
    return self.__source_type
  def GetPosition(self):
    return self.__position

  def get_L(self,P):
    if self.GetSourseType() == "ambient":
      return np.array((0,0,0))
    elif self.GetSourseType() == "point":
      L= self.GetPosition() - P
    elif self.GetSourseType() == "directional":
      L=self.GetPosition()
    
    L = L/np.linalg.norm(L)
    return np.array(L)

def ReflectRay(R, N):
  return 2*N*np.dot(N, R) - R

def ClosestIntersection(O, D, t_min, t_max):
  closest_t = np.inf
  closest_sphere = None
  for sphere in Spheres:
    [t1, t2] = IntersectRaySphere(O, D, sphere)
    if t1 < closest_t and (t_min < t1 < t_max):
      closest_t = t1
      closest_sphere = sphere
    if t2 < closest_t and (t_min < t2 < t_max):
      closest_t = t2
      closest_sphere = sphere
  return [closest_sphere, closest_t]

def IntersectRaySphere(O, D, sphere):
  C = sphere.GetCenter()
  r = sphere.GetRadius()
  OC = O - C

  k1 = np.dot(D, D)
  k2 = 2*np.dot(OC, D)
  k3 = np.dot(OC, OC) - r*r

  discriminant = k2*k2 - 4*k1*k3
  if discriminant < 0:
    return [np.inf, np.inf]

  t1 = (-k2 + np.sqrt(discriminant)) / (2*k1)
  t2 = (-k2 - np.sqrt(discriminant)) / (2*k1)
  return [t1, t2]

def TraceRay(O, D, t_min, t_max, depth):
[closest_sphere, closest_t] = ClosestIntersection(O, D, t_min, t_max)

if closest_sphere == None:
  return BACKGROUND_COLOR

P = O + closest_t*D  
N = P - closest_sphere.GetCenter() 
N = N / np.linalg.norm(N)
local_color = closest_sphere.GetColor()*ComputeLighting(P, N, -D, closest_sphere.GetSpecular())

r = closest_sphere.GetReflective()
if depth <= 0 or r <= 0:
  return local_color

R = ReflectRay(-D, N)
reflected_color = TraceRay(P, R, 0.001, np.inf, depth - 1)

return local_color*(1 - r) + reflected_color*r

def ComputeLighting(P, N, V, s):
  i = 0.0
  for light in Lights:
    if light.GetSourseType() == "ambient":
      i += light.GetIntensity()
      continue
    elif light.GetSourseType() == "point":
      L = light.GetPosition() - P
      t_max = 1
    else:
      L = light.GetPosition()
      t_max = np.inf

    [shadow_sphere, shadow_t] = ClosestIntersection(P, L, 0.001, t_max)
    if shadow_sphere != None:
        continue

    n_dot_l = np.dot(N, L)
    if n_dot_l > 0:
      i+= light.GetIntensity()*n_dot_l/(np.linalg.norm(N)*np.linalg.norm(L))
    if s != -1:
      R = 2*N*np.dot(N, L) - L
      r_dot_v = np.dot(R, V)
      if r_dot_v > 0:
        i += light.GetIntensity()*((r_dot_v/(np.linalg.norm(R)*np.linalg.norm(V)))** s)
  return i

def Render():
  Im = np.asarray(Image.new(mode='RGB', size=(Cw, Ch), color=(220, 220, 220)))
  Im2 = np.zeros(Im.shape, Im.dtype)
  Cwp = int(Cw/2)
  Chp = int(Ch/2)
  for x in range(-Cwp, Cwp):
    for y in range(-Chp, Chp):
      D = np.array([x*Vw/Cw, y*Vh/Ch, projection_plane_d])
      k = TraceRay(O, D, 0.001, np.inf, recursion_depth)
      k[0] = 255 if k[0]>255 else k[0]
      k[1] = 255 if k[1]>255 else k[1]
      k[2] = 255 if k[2]>255 else k[2]
      Im2[y+Chp][x+Cwp] = k
  display(Image.fromarray(Im2, mode='RGB'))

sphere1 = Sphere([0, -1, 3], 1.0,[240, 10, 10],500)
sphere2 = Sphere([-1, 0.8, 3], 1.0, [10, 10, 240],500)
sphere3 = Sphere([1, 0.8, 3], 1.0, [10, 240, 10],10) 
sphere4 = Sphere([0, -500, 500], 500, [240, 240, 10],1000) 

light1 = Light("ambient",(0.5,0.5,0.5), 0.2)
light2 = Light("point",(2, 1, 0),0.6)
# light2 = Light("point",(0, 0, 10),1.6)
light3 = Light("directional", (1, 4, 4), 0.2)
light4 =Light("point",(10, -10, 3),1.6)

# Spheres = [sphere1,sphere2,sphere3]
# Lights = [light1,light2,light3]
# Lights = [light4]

sp1 = Sphere([-0.6, 0, 1], 0.4,[125, 10, 10],100,0)
sp2 = Sphere([-.6,-.8, 2], 0.5, [10, 125, 10],50,0.5)
sp3 = Sphere([1.1, 0, 2], 0.6, [10, 10, 125],100,0)
sp4 = Sphere([0, 0, 5], 0.3, [240, 240, 10],0)
li1 = Light(position=(2.1, .1, 0), intensity=.7)
li2 = Light("ambient",position=np.array([1,1,1]),intensity=.2)
li3 = Light("directional", position=(1.1, -1.1, -1), intensity=.3)

# Spheres = [sp1,sp3,sp2,sp4]
# Lights = [li1,li2,li3]


s1 = Sphere([0.5, 0, 1], 0.3, [0, 240, 0],800,0.3)
s2 = Sphere([0,0.5, 1], 0.3, [20, 20, 240],800,0.2)
l1 = Light(position=(0, 0, 1), intensity=2.7)

Spheres = [s1,s2]
Lights = [l1]

Render()
