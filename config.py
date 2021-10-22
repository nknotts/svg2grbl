def in2mm(val_in):
    return val_in * 25.4


def gcode_pause(timeout_s):
    yield f'G4 P{timeout_s:.2f}'


# these functions are using by svg2gcode.py as part of gcode generation
def preamble():
    yield 'G90'


def postamble():
    yield ''
    yield 'G0 X0 Y0'


def pen_up():
    yield 'M5'
    yield from gcode_pause(0.15)


def pen_down():
    yield 'M3 S30'
    yield from gcode_pause(0.15)


CANVAS_WIDTH_X = in2mm(11)
CANVAS_WIDTH_Y = in2mm(8.5)

CANVAS_LEFT_MARGIN = in2mm(1)
CANVAS_RIGHT_MARGIN = in2mm(1)
CANVAS_TOP_MARGIN = in2mm(1)
CANVAS_BOTTOM_MARGIN = in2mm(1)

USABLE_WIDTH = CANVAS_WIDTH_X - CANVAS_LEFT_MARGIN - CANVAS_RIGHT_MARGIN
USABLE_HEIGHT = CANVAS_WIDTH_Y - CANVAS_TOP_MARGIN - CANVAS_BOTTOM_MARGIN

CANVAS_MIDPOINT_X = CANVAS_LEFT_MARGIN + USABLE_WIDTH / 2
CANVAS_MIDPOINT_Y = CANVAS_BOTTOM_MARGIN + USABLE_HEIGHT / 2
