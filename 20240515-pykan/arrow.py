"""Showcase of flying arrows that can stick to objects in a somewhat 
realistic looking way.
"""
import sys
import random
from typing import List
from collections import deque

from kan import *
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d


def create_arrow(color=None):
    vs = [(-30, 0), (0, 2), (10, 0), (0, -2)]
    # mass = 1
    # moment = pymunk.moment_for_poly(mass, vs)
    arrow_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    arrow_shape = pymunk.Poly(arrow_body, vs)
    arrow_shape.friction = 0.5
    arrow_shape.collision_type = 1
    arrow_shape.density = 0.1
    arrow_shape.filter = pymunk.ShapeFilter(group=1)
    if color:
        arrow_shape.color = color
    return arrow_body, arrow_shape


def stick_arrow_to_target(space, arrow_body, arrow_shape, target_body, position, flying_arrows):
    if "is_target" in target_body.__dict__:
        arrow_shape.color = (50, 255, 50, 255)
        hitarrow.append([
            (target_body.position.x - arrow_body.init_position.x)/100, 
            (target_body.position.y - arrow_body.init_position.y)/100, 
            ])
        hitlabel.append([arrow_body.init_velocity.x])
        dataset = {}
        dataset["train_input"] = torch.tensor(list(hitarrow)).float()
        dataset["train_label"] = torch.tensor(list(hitlabel)).float()
        dataset["test_input"] = dataset["train_input"]
        dataset["test_label"] = dataset["train_label"]
        model.train(dataset, opt="LBFGS", steps=1)
        hitrate.append(1)
    else:
        hitrate.append(0)
    
    pivot_joint = pymunk.PivotJoint(arrow_body, target_body, position)
    phase = target_body.angle - arrow_body.angle
    gear_joint = pymunk.GearJoint(arrow_body, target_body, phase, 1)
    space.add(pivot_joint)
    space.add(gear_joint)
    arrow_body.pivot_joint = pivot_joint
    arrow_body.gear_joint = gear_joint
    try:
        flying_arrows.remove(arrow_body)
    except:
        pass


def post_solve_arrow_hit(arbiter, space, data):
    # if arbiter.total_impulse.length > 300:
    a, b = arbiter.shapes
    position = arbiter.contact_point_set.points[0].point_a
    b.collision_type = 0
    b.group = 1
    other_body = a.body
    arrow_body = b.body
    space.add_post_step_callback(
        stick_arrow_to_target,
        arrow_body,
        b,
        other_body,
        position,
        data["flying_arrows"],
    )


model = KAN(width=[2,3,3,3,1], grid=5, k=3, seed=0)
model.is_trained = False
hitrate = deque(maxlen=100)
hitarrow = deque(maxlen=32)
hitlabel = deque(maxlen=32)

width, height = 1200, 600


def main():
    ### PyGame init
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Arial", 16)

    ### Physics stuff
    space = pymunk.Space()
    space.gravity = 0, 1000
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # walls - the left-top-right walls
    static: List[pymunk.Shape] = [
        pymunk.Segment(space.static_body, (0, 550), (1200, 550), 5),
    ]

    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    b2.is_target = True
    vs = [(0, 0), (35, 0), (35, 10), (0, 10)]
    target_bad = pymunk.Poly(b2, vs)
    b2.position = 700, 535

    for s in static:
        s.friction = 1.0
        s.group = 1
    space.add(b2, target_bad, *static)

    # "Cannon" that can fire arrows
    cannon_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    cannon_shape = pymunk.Circle(cannon_body, 25)
    cannon_shape.sensor = True
    cannon_shape.color = (255, 50, 50, 255)
    cannon_body.position = 100, 200
    direction = 1
    space.add(cannon_body, cannon_shape)

    arrow_queue = deque()
    arrow_body, arrow_shape = create_arrow()
    space.add(arrow_body, arrow_shape)

    flying_arrows: List[pymunk.Body] = []
    handler = space.add_collision_handler(0, 1)
    handler.data["flying_arrows"] = flying_arrows
    handler.post_solve = post_solve_arrow_hit

    start_time = 0
    activate_time = -5001
    while running:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                start_time = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                end_time = pygame.time.get_ticks()

        if pygame.mouse.get_pressed()[0]:
            current_time = pygame.time.get_ticks()
            if current_time - start_time > 100:
                start_time = pygame.time.get_ticks()

                if random.randint(1, 100) <= sum(hitrate):
                    activate_time = current_time

                if current_time - activate_time < 2000:
                    power = float(model(torch.tensor([[
                        (b2.position.x - cannon_body.position.x)/100, 
                        (b2.position.y - cannon_body.position.y)/100
                        ]]).float()).detach().numpy()[0][0]) + np.random.normal(0, 10)
                    color = (255, 50, 50, 255)
                else:
                    power = random.randint(200, 1000)
                    color = None
                # power = max(min(diff, 1000), 10)
                impulse = power * Vec2d(1, 0)
                arrow_body.body_type = pymunk.Body.DYNAMIC
                arrow_body.mass = 1.0
                arrow_body.apply_impulse_at_world_point(impulse, arrow_body.position)
                arrow_body.init_velocity = arrow_body.velocity
                arrow_body.init_position = arrow_body.position

                # space.add(arrow_body)
                flying_arrows.append(arrow_body)
                arrow_queue.append((arrow_body, arrow_shape))

                arrow_body, arrow_shape = create_arrow(color)
                space.add(arrow_body, arrow_shape)


        if len(arrow_queue) > 100:
            arrow_body_remove, arrow_shape_remove = arrow_queue.popleft()
            if "pivot_joint" in arrow_body_remove.__dict__:
                space.remove(arrow_body_remove.pivot_joint, arrow_body_remove.gear_joint)
            space.remove(arrow_body_remove, arrow_shape_remove)
        # keys = pygame.key.get_pressed()


        
        if cannon_body.position.y > 350:
            direction = -1
        elif cannon_body.position.y < 150:
            direction = 1

        speed = 2.5
        cannon_body.position += Vec2d(0, direction) * speed

        arrow_body.position = cannon_body.position + Vec2d(
            cannon_shape.radius + 40, 0
        )

        for flying_arrow in flying_arrows:
            flight_direction = Vec2d(*flying_arrow.velocity)
            flying_arrow.angle = flight_direction.angle

        ### Clear screen
        screen.fill(pygame.Color("black"))

        ### Draw stuff
        space.debug_draw(draw_options)

        # Info and flip screen
        screen.blit(
            font.render("fps: " + str(clock.get_fps()), True, pygame.Color("white")),
            (0, 0),
        )
        screen.blit(
            font.render(
                "Press and hold the left mouse button to shoot",
                True,
                pygame.Color("darkgrey"),
            ),
            (5, height - 35),
        )
        screen.blit(
            font.render("Press ESC or Q to quit", True, pygame.Color("darkgrey")),
            (5, height - 20),
        )

        pygame.display.flip()

        ### Update physics
        fps = 60
        dt = 1.0 / fps
        space.step(dt)

        clock.tick(fps)


if __name__ == "__main__":
    sys.exit(main())