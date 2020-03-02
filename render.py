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
viewpoint = np.array([0, 0, 0])
BACKGROUND_COLOR = np.array([0, 0, 0])
recursion_depth = 3


class Sphere:
  def __init__(self, center=np.array([0, 0, 0]), radius=1.0, color=np.array(BACKGROUND_COLOR), specular = -1, reflective = 0, refractive = 0):
    self.__center = center
    self.__radius = radius
    self.__color=np.array(color)
    self.__specular = specular
    self.__reflective = reflective
    self.__refractive = refractive

  # def getRadius(self):
  #   return self.__radius
  # def getCenter(self):
  #   return self.__center
  def getColor(self):
    return self.__color
  def getSpecular(self):
    return self.__specular
  def getReflective(self):
    return self.__reflective
  def getRefractive(self):
    return self.__refractive
  def getNorm(self, point):
    norm = point - self.__center 
    norm = norm / np.linalg.norm(norm)
    return norm

  def intersectRaySphere(self, viewpoint, view_direction):
    # a*x*x + b*x + c = 0
    OC = viewpoint - self.__center

    a = np.dot(view_direction, view_direction)
    b = 2*np.dot(OC, view_direction)
    c = np.dot(OC, OC) - self.__radius ** 2

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return [np.inf, np.inf]

    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    return [t1, t2]

  def reflectRay(self, point, view_direction):
    norm = self.getNorm(point)
    return 2*norm*np.dot(norm, view_direction) - view_direction

  def getRefractRay(self, point, view_direction):
    norm = self.getNorm(point)
    eta = 1./self.__refractive
    cos_theta = -np.dot(norm, view_direction)
    if cos_theta < 0:
      cos_theta = -cos_theta
      norm = -norm
      eta = -eta
    k = 1. - eta*eta*(1.0-cos_theta*cos_theta);
    if k < 0:
      return None
    # ray_dir = normalize( eta*view_direction + (eta*cos_theta - np.sqrt(k))*norm);
    ray_dir = eta*view_direction + (eta*cos_theta - np.sqrt(k))*norm;
    return np.array(ray_dir)

  def refractRay(self, point, view_direction):
    new_point = None
    ray_dir = self.getRefractRay(point, view_direction)
    if ray_dir is not None:
      [t1, t2] = self.intersectRaySphere(view_direction, ray_dir)
      if [t1, t2] != [np.inf,np.inf]:
        tmp = view_direction + t1*ray_dir
        new_point = tmp if tmp.all() != point.all() else view_direction + t2*ray_dir
        ray_dir = self.getRefractRay(new_point, ray_dir)
      else:
        ray_dir = None

    return [new_point, ray_dir]
    

class Light:
  def __init__(self,source_type="point",position=np.array([0.5,0.5,0.5]),intensity=0.5):
    self.__source_type=source_type
    self.__position= np.array(position)
    self.__intensity = intensity
      
  def getIntensity(self):
    return self.__intensity
  # def getSourseType(self):
  #   return self.__source_type

  def getPosition(self, point):
    if self.__source_type == "ambient":
      return None
    elif self.__source_type == "point":
      return self.__position - point
    else:
      return self.__position

  def getTMax(self):
    if self.__source_type == "ambient":
      return None
    elif self.__source_type == "point":
      return 1
    else:
      return np.inf


def closestIntersection(viewpoint, view_direction, t_min, t_max):
  closest_t = np.inf
  closest_sphere = None
  for sphere in spheres:
    [t1, t2] = sphere.intersectRaySphere(viewpoint, view_direction)
    if t1 < closest_t and (t_min < t1 < t_max):
        closest_t = t1
        closest_sphere = sphere
    if t2 < closest_t and (t_min < t2 < t_max):
        closest_t = t2
        closest_sphere = sphere
  return [closest_sphere, closest_t]

def traceRay(viewpoint, view_direction, t_min, t_max, depth):
  [closest_sphere, closest_t] = closestIntersection(viewpoint, view_direction, t_min, t_max)

  if closest_sphere == None:
    return BACKGROUND_COLOR

  point = viewpoint + closest_t*view_direction
  local_color = closest_sphere.getColor()*computeLighting(point, closest_sphere.getNorm(point), -view_direction, closest_sphere.getSpecular())

  reflective = closest_sphere.getReflective()
  refractive = closest_sphere.getRefractive()
  if depth <= 0:
    return local_color

  if reflective > 0:
    reflected_ray = closest_sphere.reflectRay(point, -view_direction)
    reflected_color = traceRay(point, reflected_ray, 0.001, np.inf, depth - 1)
  if refractive > 0:
    [point_new, refracted_ray] = closest_sphere.refractRay(point, -view_direction)
    if refracted_ray is not None:
      refracted_color = traceRay(point_new, refracted_ray, 0.001, np.inf, depth - 1)
    else:
      refractive = 0
  
  if reflective > 0 and refractive > 0:
    return local_color*(1-reflective-refractive) + reflected_color*reflective + refracted_color*refractive
  elif reflective > 0:
    return local_color*(1-reflective) + reflected_color*reflective
  elif refractive > 0:
    return local_color*(1-refractive) + refracted_color*refractive
  else:
    return local_color

def computeLighting(point, norm_sphere, view_direction, specular):
  intensity = 0.0
  for light in lights:
    light_position = light.getPosition(point)
    if light_position is None:
      intensity += light.getIntensity()
      continue

    [shadow_sphere, shadow_t] = closestIntersection(point, light_position, 0.001, light.getTMax())
    if shadow_sphere != None:
      continue

    n_dot_l = np.dot(norm_sphere, light_position)
    if n_dot_l > 0:
      intensity += light.getIntensity()*n_dot_l/(np.linalg.norm(norm_sphere)*np.linalg.norm(light_position))
    if specular != -1:
      R = 2*norm_sphere*np.dot(norm_sphere, light_position) - light_position
      r_dot_v = np.dot(R, view_direction)
      if r_dot_v > 0:
        intensity += light.getIntensity()*((r_dot_v/(np.linalg.norm(R)*np.linalg.norm(view_direction)))** specular)
  return intensity

def normColor(point_color):
  point_color[0] = 255 if point_color[0]>255 else point_color[0]
  point_color[1] = 255 if point_color[1]>255 else point_color[1]
  point_color[2] = 255 if point_color[2]>255 else point_color[2]
  return point_color

def render():
  Im = np.asarray(Image.new(mode='RGB', size=(Cw, Ch), color=(220, 220, 220)))
  Im2 = np.zeros(Im.shape, Im.dtype)
  Cwp = int(Cw/2)
  Chp = int(Ch/2)
  for x in range(-Cwp, Cwp):
    for y in range(-Chp, Chp):
      view_direction = np.array([x*Vw/Cw, y*Vh/Ch, projection_plane_d])
      point_color = traceRay(viewpoint, view_direction, 0.001, np.inf, recursion_depth)
      Im2[y+Chp][x+Cwp] = normColor(point_color)
  display(Image.fromarray(Im2, mode='RGB'))

sphere1 = Sphere([0, -1, 3], 1.0,[240, 10, 10],500)
sphere2 = Sphere([-1, 0.8, 3], 1.0, [10, 10, 240],500)
sphere3 = Sphere([1, 0.8, 3], 1.0, [10, 240, 10],10) 
sphere4 = Sphere([0, -500, 500], 500, [240, 240, 10],1000) 

light1 = Light("ambient",(0.5,0.5,0.5), 0.2)
light2 = Light("point",(2, 1, 0),0.6)
# light2 = Light("point",(0, 0, 10),1.6)
light3 = Light("directional", (1, 4, 4), 0.2)
light4 = Light("point",(10, -10, 3),1.6)

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


s1 = Sphere([0.5, 0, 1], 0.3, [0, 240, 0],800,0.3,0.5)
s2 = Sphere([0,0.5, 1], 0.3, [20, 20, 240],800,0.2)
l1 = Light(position=(0, 0, 1), intensity=2.7)

s3 = Sphere([0, 0, 1], 0.2, [0, 240, 0],800,0.,55)
s4 = Sphere([-0.3, -0.3, 1], 0.2, [20, 20, 240],800,0.,0.5)
l2 = Light(position=(0.5, 0.5, 1), intensity=2.7)
l3 = Light(position=(0.5, -0.5, 1), intensity=2.7)

spheres = [s1,s2]
lights = [l1]

render()
