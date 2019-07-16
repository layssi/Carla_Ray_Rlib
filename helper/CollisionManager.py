import math
import weakref

import carla

import queue

# from helper.CarlaDebug import get_actor_display_name


class CollisionManager(object):
    def __init__(self, parent_actor, synchronous_mode=True):
        self.sensor = None
        self.intensity = False
        self._parent = parent_actor
        self.synchronous_mode = synchronous_mode
        self.world = self._parent.get_world()
        self.bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.world = self._parent.get_world()
        self.sensor = self.world.spawn_actor(
            self.bp, carla.Transform(), attach_to=self._parent
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        if not self.synchronous_mode:
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: CollisionManager._on_collision(weak_self, event)
            )
        else:
            self.collision_queue = None
            self.collision_queue = queue.Queue()
            self.sensor.listen(self.collision_queue.put)

    def read_collision_queue(self):
        weak_self = weakref.ref(self)
        if not self.synchronous_mode:
            return self.intensity
        else:
            try:
                CollisionManager._on_collision(
                    weak_self, self.collision_queue.get(False)
                )
            except:
                # print("We could not get collision sensor")
                # Ignore empty Que
                pass

    def destroy_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
            self.intensity = False

    def get_collision_data(self):
        if self.intensity is not False:
            return True
        else:
            return False

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        # actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        self.intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
