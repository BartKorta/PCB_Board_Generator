
class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y=y

    def to_str(self):
        return "(%d,%d)" % (self.get_x(), self.get_y())

    def to_str2(self):
        return "%d %d" % (self.get_x(), self.get_y())

    @staticmethod
    def from_str_to_point(string):
        x = string.split(" ")
        p = Point(int(x[0]), int(x[1]))
        return p

    def increase_x(self, step):
        self.x += step

    def increase_y(self, step):
        self.y += step

    def is_inside(self, max_x, max_y):
        if self.get_x() < 0 or self.get_x() >= max_x:
            return False
        elif self.get_y() < 0 or self.get_y() >= max_y:
            return False
        return True

    def get_coordinates(self):
        return self.x, self.y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False

    def __ne__(self, other):
        if self.x != other.x or self.y != other.y:
            return True
        else:
            return False

class Connection:

    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def get_points_coordinates(self):
        return self.point1.get_coordinates(), self.point2.get_coordinates()

    def get_start_point(self):
        return self.point1

    def get_end_point(self):
        return self.point2

    def __str__(self):
        return 'Connection between Point1: ( %d , %d ) and Point2: ( %d , %d )' \
               % (self.point1.get_x(), self.point1.get_y(), self.point2.get_x(), self.point2.get_y())


class Data:

    def __init__(self):
        self.x_dim = 0
        self.y_dim = 0
        self.connections = list()
        self.no_conn = 0

    def get_x_dim(self):
        return self.x_dim

    def get_y_dim(self):
        return self.y_dim

    def get_connections(self):
        return self.connections

    def get_number_of_connections(self):
        return self.no_conn

    def add_connection(self, point1, point2):
        new_connection = Connection(point1, point2)
        self.connections.append(new_connection)
        self.no_conn += 1

    def add_connection_with_coordinates(self, x1, y1, x2, y2):
        point1 = Point(x1, y1)
        point2 = Point(x2, y2)
        self.add_connection(point1, point2)

    def set_dimensions(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __str__(self):
        result = 'Data:\n   Dimensions: %d x %d' % (self.x_dim, self.y_dim)
        for i in self.connections:
            result += '\n       ' + str(i)

        return result

    def set_data_from_filename(self, filename):
        file = open(filename, "r")

        line = file.readline()
        dims = line.split(';')
        self.set_dimensions(int(dims[0]), int(dims[1]))

        while line != '':
            line = file.readline()
            coordns = line.split(';')
            if len(coordns) == 4:
                self.add_connection_with_coordinates(int(coordns[0]), int(coordns[1]), int(coordns[2]), int(coordns[3]))

