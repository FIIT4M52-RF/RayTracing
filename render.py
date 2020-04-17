import PIL
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from random import randint

BACKGROUND_COLOR = np.array([0, 0, 0])

class Light:
    def __init__(self, intensity=0.5, position=np.array([0.5,0.5,0.5])):
        self.position = np.array(position)
        self.intensity = intensity
        self.max_t = None
        
    def getIntensity(self):
        return self.intensity

    def getPosition(self, point):
        return self.position

    def maxT(self):
        return self.max_t


class LightAmbient(Light):
    def __init__(self, intensity=0.5):               
        self.intensity = intensity
        self.position = None
        self.max_t = None


class LightDerectional(Light):
    def __init__(self, intensity=0.5, position=np.array([0.5,0.5,0.5])):        
        self.intensity = intensity
        self.position = np.array(position)
        self.max_t = np.inf


class LightPoint(Light):
    def __init__(self, intensity=0.5, position=np.array([0.5,0.5,0.5])):        
        self.intensity = intensity
        self.position = np.array(position)
        self.max_t = 1

    def getPosition(self, point):
        return self.position - point


class Sphere:
    def __init__(self, center=np.array([0, 0, 0]), radius=1.0, 
                 color=np.array(BACKGROUND_COLOR), specular = None, 
                 reflective = None, transparensy = None):
        self.center = center
        self.radius = radius
        self.color=np.array(color)

        self.specular = specular
        self.reflective = reflective
        self.transparensy = transparensy
        
        self.point = None
        self.norm = None
        
    def setPoint(self, viewpoint, closest_t, view_direction):
        self.point = viewpoint + closest_t*view_direction
        norm = self.point - self.center 
        self.norm = norm / np.linalg.norm(norm)

    def getColor(self):
        return self.color   

    def getSpecular(self):
        return self.specular

    def getReflective(self):
        return self.reflective

    def getRefractive(self):
        return self.refractive

    def getTransparensy(self):
        return self.transparensy

    def getNorm(self):
        return self.norm

    def getPoint(self):
        return self.point

    def intersectRaySphere(self, viewpoint, view_direction):
        # a*x*x + b*x + c = 0
        OC = viewpoint - self.center

        a = np.dot(view_direction, view_direction)
        b = 2*np.dot(OC, view_direction)
        c = np.dot(OC, OC) - self.radius ** 2

        discriminant = b*b - 4*a*c
        if discriminant < 0:
            [t1, t2] =  [np.inf, np.inf]
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
        return [t1, t2] 

    def reflectRay(self, view_direction):
        return 2*self.norm*np.dot(self.norm, view_direction) - view_direction

    def transparensyRay(self, view_direction):
        [t1, t2] = self.intersectRaySphere(self.point, view_direction)
        new_point = self.point + t1*view_direction
        if new_point.all() == self.point.all():
           new_point =  self.point + t2*view_direction
        return new_point


class Render:
    def __init__(self, spheres, lights):
        self.spheres = spheres
        self.lights = lights

    def normColor(self, point_color):
        point_color[0] = 255 if point_color[0]>255 else point_color[0]
        point_color[1] = 255 if point_color[1]>255 else point_color[1]
        point_color[2] = 255 if point_color[2]>255 else point_color[2]
        return point_color

    def show(self, Cw = 200, Ch = 200, Vw = 1, Vh = 1, projection_plane_d = 1, 
             viewpoint = np.array([0, 0, 0]), recursion_depth = 3):
      
        Im = np.asarray(Image.new(mode='RGB', size=(Cw, Ch), 
                                  color=(220, 220, 220)))
        Im2 = np.zeros(Im.shape, Im.dtype)
        Cwp = int(Cw/2)
        Chp = int(Ch/2)
        for x in range(-Cwp, Cwp):
            for y in range(-Chp, Chp):
                view_direction = np.array([x*Vw/Cw, y*Vh/Ch, 
                                           projection_plane_d])
                point_color = self.traceRay(viewpoint, view_direction, 0.001, 
                                            np.inf, recursion_depth)
                Im2[y+Chp][x+Cwp] = self.normColor(point_color)
        display(Image.fromarray(Im2, mode='RGB'))

    def closestIntersection(self, viewpoint, view_direction, t_min, t_max):
        closest_t = np.inf
        closest_sphere = None
        for sphere in self.spheres:
            [t1, t2] = sphere.intersectRaySphere(viewpoint, view_direction)
            if t1 < 0.01 or t2 < 0.01:
                continue
            if t1 < closest_t and (t_min < t1 < t_max):
                closest_t = t1
                closest_sphere = sphere
            if t2 < closest_t and (t_min < t2 < t_max):
                closest_t = t2
                closest_sphere = sphere
        if closest_sphere is not None:
            point = viewpoint + closest_t*view_direction
            closest_sphere.setPoint(viewpoint, closest_t, view_direction)
        return closest_sphere

    def traceRay(self, viewpoint, view_direction, t_min, t_max, depth):
        closest_sphere = self.closestIntersection(viewpoint, view_direction, 
                                                  t_min, t_max)
        if closest_sphere == None:
            return BACKGROUND_COLOR

        local_color = self.computeLighting(closest_sphere, -view_direction)

        if depth <= 0:
            return local_color

        reflective = closest_sphere.getReflective()
        if reflective:
            reflected_color = self.traceRay(closest_sphere.getPoint(), 
                              closest_sphere.reflectRay(-view_direction), 
                              0.001, np.inf, depth - 1)
            local_color = (local_color*(1 - reflective)
                          + reflected_color*reflective)

        transparensy = closest_sphere.getTransparensy()
        if transparensy:
            transparensy_point = closest_sphere.transparensyRay(-view_direction)
            transparensy_color = self.traceRay(transparensy_point, 
                                               closest_sphere.getPoint(), 
                                               0.001, np.inf, depth - 1)
            local_color = local_color + transparensy_color*transparensy        

        return local_color

    def computeLighting(self, sphere, view_direction):
        intensity = 0.0
        shadow_color = np.array([0, 0, 0])
        for light in self.lights:
            light_position = light.getPosition(sphere.getPoint())
            light_intensity = light.getIntensity()
            if light_position is None:
                intensity += light_intensity
                continue
            intensity_local = 0.0
            shadow_transparensy = 1
            shadow_sphere = self.closestIntersection(sphere.getPoint(), 
                                                     light_position, 0.001, 
                                                     light.maxT())
            if shadow_sphere != None:
                shadow_transparensy = shadow_sphere.getTransparensy()
                if shadow_transparensy == None:
                    continue

            sphere_norm = sphere.getNorm()
            n_dot_l = np.dot(sphere_norm, light_position)
            if n_dot_l > 0:
                intensity_local += (light_intensity
                                    *n_dot_l/np.linalg.norm(light_position))
            if sphere.getSpecular():
                R = (2*sphere_norm*np.dot(sphere_norm, light_position) 
                    - light_position)
                r_dot_v = np.dot(R, view_direction)
                if r_dot_v > 0:
                    intensity_local += (light_intensity*((r_dot_v
                                       /(np.linalg.norm(R)
                                       *np.linalg.norm(view_direction)))
                                       ** sphere.getSpecular()))
            if shadow_sphere != None:
                shadow_intensity = intensity_local*shadow_transparensy
                shadow_color = (shadow_intensity*shadow_sphere.getColor()
                               + shadow_color)
            intensity += intensity_local * shadow_transparensy
        return shadow_color + sphere.getColor()*intensity

if __name__ == "__main__":
    sp1 = Sphere([-0.6, 0, 1], 0.4, [125, 10, 10], 100, 0, 0.4)
    sp2 = Sphere([-.6,-.8, 2], 0.5, [10, 125, 10], 50, 0.5)
    sp3 = Sphere([1.1, 0, 2], 0.6, [10, 10, 125], 100, 0)
    sp4 = Sphere([0, 0, 5], 0.3, [240, 240, 10], 0)
    li1 = LightPoint(intensity=.7, position=(2.1, .1, 0))
    li2 = LightAmbient(intensity=.2)
    li3 = LightDerectional(intensity=.3, position=(1.1, -1.1, -1))

    spheres = [sp1,sp3,sp2,sp4]
    lights = [li1,li2,li3]

    render = Render(spheres, lights)
    render.show(Cw = 1000, Ch = 1000)
