
class conv2eve(object):
	""" Transform a matrix into event type:
                .t -> time
                .x and .y -> spatial position
                .p -> polarity"""

	def __init__(self, t, x, y, p):
		self.t = t
		self.x = x
		self.y = y
		self.p = p
