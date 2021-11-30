#!/usr/bin/env python3

import math
import argparse
import logging
import time

import svgpathtools.document
import svgpathtools.path

import numpy

from config import *


def extract_points(path):
    def transform(x, y):
        return numpy.array(x), -numpy.array(y)

    x = []
    y = []

    last_x = None
    last_y = None

    for segment in path:
        if segment.start.real != last_x or segment.start.imag != last_y:
            if x:
                yield transform(x, y)
            x = []
            y = []

        last_x = segment.end.real
        last_y = segment.end.imag

        if isinstance(segment, svgpathtools.path.Line):
            x.append(segment.start.real)
            x.append(segment.end.real)
            y.append(segment.start.imag)
            y.append(segment.end.imag)
        elif isinstance(segment, svgpathtools.path.CubicBezier) or isinstance(segment, svgpathtools.path.QuadraticBezier):
            arc_len = segment.length()
            requested_num_segment = math.ceil(
                arc_len / BEZIER_DISCRETIZATION_LENGTH_MM)

            poly = segment.poly()
            for t in numpy.linspace(0, 1, min(requested_num_segment, MAX_BEZIER_DISCRETIZATION_POINTS)):
                v = poly(t)
                x.append(v.real)
                y.append(v.imag)
        else:
            raise Exception

    yield transform(x, y)


def get_bounds(paths):
    bounding_boxes = [path.bbox() for path in paths]

    class Bounds(object):
        def __init__(self):
            self.min_x = min(b[0] for b in bounding_boxes)
            self.max_x = max(b[1] for b in bounding_boxes)
            self.min_y = min(b[2] for b in bounding_boxes)
            self.max_y = max(b[3] for b in bounding_boxes)
            self.mid_x = (self.max_x + self.min_x) / 2
            self.mid_y = (self.max_y + self.min_y) / 2
            self.width = self.max_x - self.min_x
            self.height = self.max_y - self.min_y
            self.is_landscape = True if self.width >= self.height else False

        def __repr__(self):
            return f'Bounds(min_x:{self.min_x:.2f}, max_x:{self.max_x:.2f}, width:{self.width:.2f}, min_y:{self.min_y:.2f}, max_y:{self.max_y:.2f}, height:{self.height:.2f})'

    return Bounds()


def rotate_paths(paths, angle_deg, bounds):
    return [p.rotated(-angle_deg, complex(bounds.min_x, bounds.min_y)) for p in paths]


def scale_paths(paths, scale, bounds):
    return [p.scaled(scale, scale, complex(bounds.min_x, bounds.min_y)) for p in paths]


def translate_paths(paths, x, y):
    return [p.translated(complex(x, y)) for p in paths]


def generate_paths(fname):
    is_canvas_landscape = True if USABLE_WIDTH >= USABLE_HEIGHT else False

    doc = svgpathtools.document.Document(fname)
    paths = doc.paths()
    bounds = get_bounds(paths)

    document_rotation_deg = MANUAL_ROTATION_DEG
    if AUTO_ROTATE:
        document_rotation_deg = 0 if is_canvas_landscape == bounds.is_landscape else 90

    paths = rotate_paths(paths, document_rotation_deg, bounds)
    bounds = get_bounds(paths)

    # auto scale to usable bounds
    scale_x = USABLE_WIDTH / bounds.width
    scale_y = USABLE_HEIGHT / bounds.height
    min_scale = min(scale_x, scale_y)
    paths = scale_paths(paths, min_scale, bounds)
    bounds = get_bounds(paths)

    shift_x = CANVAS_MIDPOINT_X - bounds.mid_x
    shift_y = CANVAS_MIDPOINT_Y - bounds.mid_y

    return translate_paths(paths, shift_x, shift_y - 2 * CANVAS_MIDPOINT_Y)


def gcode_move(x, y):
    yield f'G0 X{x:.2f} Y{y:.2f}'


def gcode_arc(x_center, y_center, radius, theta_start_deg=0, arc_travel_deg=360, feedrate=30000, cw=True):
    theta_end_deg = theta_start_deg + arc_travel_deg
    theta_start_rad = theta_start_deg * math.pi / 180
    theta_end_rad = theta_end_deg * math.pi / 180

    x_start = x_center+(radius*math.cos(theta_start_rad))
    y_start = y_center+(radius*math.sin(theta_start_rad))
    x_end = x_center+(radius*math.cos(theta_end_rad))
    y_end = y_center+(radius*math.sin(theta_end_rad))
    i_term = (x_center-(radius*math.cos(theta_start_rad)))-x_center
    j_term = (y_center-(radius*math.sin(theta_start_rad)))-y_center

    yield 'G0 X{:.3f} Y{:.3f}'.format(x_start, y_start)

    arc_code = 'G2' if cw else 'G3'
    arc_code += ' X{:.3f} Y{:.3f} I{:.3f} J{:.3f} F{:.3f}'.format(
        x_end, y_end, i_term, j_term, feedrate)
    yield arc_code


def get_segments(fname):
    class Segment(object):
        def __init__(self, x, y):
            assert len(x) == len(y)
            self.size = len(x)
            self.x = x
            self.y = y

        def __repr__(self):
            return f'Segment(len={len(self.x)})'

        def reversed(self):
            return Segment(self.x[::-1], self.y[::-1])

        def distanceto(self, other):
            x = other.x[0] - self.x[-1]
            y = other.y[0] - self.y[-1]
            return math.sqrt((x*x) + (y*y))

    def paths():
        generated_paths = generate_paths(fname)
        for path_idx, path in enumerate(generated_paths):
            logging.info(f"Generating Path {path_idx+1}/{len(generated_paths)}")
            for x, y in extract_points(path):
                yield Segment(x, y)

    return [x for x in paths()]


def get_total_distance(segments):
    return sum(segments[i-1].distanceto(segments[i]) for i in range(1, len(segments)))


def optimize_segments(segments):
    segments = segments.copy()

    new_order = []
    new_order.append(segments.pop(0))

    l = len(segments)

    while len(new_order) <= l:
        shortest = float('Inf')
        last = new_order[-1]

        for segment in segments:
            d = last.distanceto(segment)
            d2 = last.distanceto(segment.reversed())

            if d < shortest:
                shortest = d
                selection = segment
                reverse = False

            if d2 < shortest:
                shortest = d2
                selection = segment
                reverse = True

        new_order.append(selection if reverse else selection.reversed())
        segments.remove(selection)

    return new_order


def generate_gcode(fname):
    t0 = time.time()

    yield from preamble()

    segments = get_segments(fname)
    total_distance = get_total_distance(segments)

    segments_optimized = optimize_segments(segments)
    total_optimized_distance = get_total_distance(segments_optimized)
    logging.info(f'Original Path Travel: {total_distance:.1f} mm, Optimized Path Travel: {total_optimized_distance:.1f}')

    if total_distance < total_optimized_distance:
        logging.warning("Using original path since it is shorter than optimized path")
        segments_optimized = segments 

    for segment in segments_optimized:
        yield ''
        yield from gcode_move(segment.x[0], segment.y[0])
        yield from pen_down()
        for idx in range(1, segment.size):
            yield from gcode_move(segment.x[idx], segment.y[idx])
        yield from pen_up()

    yield from postamble()

    tend = time.time()
    logging.info(f'Completed g-code generation in {tend-t0:.1f} seconds')


def svg2gcode(input_fname, output_fname):
    if output_fname is None:
        import os
        import os.path
        output_fname = os.path.join(
            os.getcwd(),
            f'{os.path.splitext(os.path.basename(input_fname))[0]}.gcode')

    with open(output_fname, 'w') as f:
        for gcode in generate_gcode(input_fname):
            f.write(f'{gcode}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fname')
    parser.add_argument('-o', '--output_fname', required=False)
    parser.add_argument(
        "-l",
        "--log-level",
        default='info',
        help='Log message level - default: "info"',
        choices=['debug', 'info', 'warning', 'error', 'critical']
    )

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d | %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(args.log_level.upper())
    svg2gcode(args.input_fname, args.output_fname)
