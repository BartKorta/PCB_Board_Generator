import random
import math
from random import choices, choice
import copy
from data_import import Data, Connection, Point
import matplotlib.pyplot as plt
COLORS = ['black','green','red','blue','yellow','red','grey','orange','purple']
class Gene:

    def __init__(self):
        self.direction = 0
        self.step = 0

    def get_direction(self):
        return self.direction

    def get_step(self):
        return self.step

    def randomize_gene(self, max_step):
        self.direction = random.randint(0, 3)
        self.step = random.randint(1, max_step)

    def set_params(self, direction, step):
        self.direction = direction
        self.step = step

    def set_step(self, step):
        self.step = step

    @staticmethod
    def get_opp_direction(dir):
        if dir == 0:
            return 2
        elif dir == 1:
            return 3
        elif dir == 2:
            return 0
        else:
            return 1

    @staticmethod
    def get_name_of_direction(direction):
        name = ''
        if direction == 0:
            name = 'Up'
        elif direction == 1:
            name = 'Right'
        elif direction == 2:
            name = 'Down'
        else:
            name = 'Left'
        return name

    @staticmethod
    def get_available_direction_after(prev_direction):
        list_available = list()
        if prev_direction == (prev_direction, 0) == 0:
            list_available.append(1)
            list_available.append(3)
        else:
            list_available.append(0)
            list_available.append(2)


class Genotype:

    def __init__(self):
        self.genes = list()

    def add_gene(self, gene):
        self.genes.append(gene)

    def get_genes(self):
        return self.genes

    def generate_genotype(self, start_point, end_point, max_x, max_y):
        current_point = start_point
        prev_dir = -1
        while current_point.__ne__(end_point):
            new_gene = Genotype.next_step_from_current(current_point, end_point, prev_dir, max_x-1, max_y-1)
            prev_dir = new_gene.get_direction() % 2
            self.add_gene(new_gene)
            current_point = Genotype.recognize_current_point(current_point, new_gene.get_direction(),
                                                             new_gene.get_step())

    def get_path_length(self):
        total = 0
        for i in self.get_genes():
            total += i.get_step()
        return total

    def get_path_len_outside(self, start_point, max_x, max_y):
        total = 0
        current_point = start_point
        for i in self.get_genes():
            step = i.get_step()
            while step > 0:
                next_point = Genotype.recognize_current_point(current_point, i.get_direction(), 1)
                step -= 1
                if not(next_point.is_inside(max_x, max_y)) or not(current_point.is_inside(max_x, max_y)):
                    total += 1
                current_point = next_point
        return total

    def get_number_of_segments_outside(self, start_point, max_x, max_y):
        total = 0
        current_point = start_point
        for i in self.get_genes():
            next_point = Genotype.recognize_current_point(current_point, i.get_direction(), i.get_step())
            if not (next_point.is_inside(max_x, max_y)) or not (current_point.is_inside(max_x, max_y)):
                total += 1
            current_point = next_point
        return total

    def get_number_of_segments(self):
        return len(self.get_genes())

    def get_list_of_temp_points(self, start_point):
        temp = list()
        temp.append(start_point)
        current_point = start_point
        for i in self.get_genes():
            step = i.get_step()
            while step > 0:
                current_point = Genotype.recognize_current_point(current_point, i.get_direction(), 1)
                temp.append(current_point)
                step -= 1
        return temp

    @staticmethod
    def next_step_from_current(current, end, prev_dir, max_x, max_y):

        if current.get_x() == end.get_x():
            new_gene = Gene()
            if prev_dir != 0:
                rand_step = random.randint(1, abs(current.get_y() - end.get_y()))
                if current.get_y() > end.get_y():
                    new_dir = 0
                else:
                    new_dir = 2
                new_gene.set_params(new_dir, rand_step)
            else:
                rand_dir = 2 * random.randint(0, 1) + 1
                if rand_dir == 1:
                    # rand_step = random.randint(1, max_x - current.get_x()+1)
                    rand_step = 1
                else:
                    #rand_step = random.randint(1, current.get_x())
                    rand_step = 1
                new_gene.set_params(rand_dir, rand_step)
            return new_gene

        elif current.get_y() == end.get_y():
            new_gene = Gene()
            if prev_dir != 1:
                rand_step = random.randint(1, abs(current.get_x() - end.get_x()))
                if current.get_x() > end.get_x():
                    new_dir = 3
                else:
                    new_dir = 1
                new_gene.set_params(new_dir, rand_step)
            else:
                rand_dir = 2 * random.randint(0, 1)
                if rand_dir == 0:
                    #rand_step = random.randint(1, current.get_y())
                    rand_step = 1
                else:
                    #rand_step = random.randint(1, max_y - current.get_y()+1)
                    rand_step = 1
                new_gene.set_params(rand_dir, rand_step)
            return new_gene

        else:
            available = list()
            if current.get_x() - end.get_x() > 0 and prev_dir != 1:
                available.append(3)
            if current.get_x() - end.get_x() < 0 and prev_dir != 1:
                available.append(1)
            if current.get_y() - end.get_y() > 0 and prev_dir != 0:
                available.append(0)
            if current.get_y() - end.get_y() < 0 and prev_dir != 0:
                available.append(2)
            rand_dir = available[random.randint(0, len(available) - 1)]

            if rand_dir % 2 == 0:
                rand_step = random.randint(1, abs(current.get_y() - end.get_y()))
            else:
                rand_step = random.randint(1, abs(current.get_x() - end.get_x()))
            new_gene = Gene()
            new_gene.set_params(rand_dir, rand_step)
            return new_gene

    @staticmethod
    def recognize_current_point(previous_point, direction, step):
        following_point = Point(previous_point.get_x(), previous_point.get_y())
        if direction == 0:
            following_point.increase_y(-step)
        elif direction == 1:
            following_point.increase_x(step)
        elif direction == 2:
            following_point.increase_y(step)
        else:
            following_point.increase_x(-step)
        return following_point

    def mutate2(self, start_point, end_point):
        len_genes = len(self.genes)
        where = 0
        if len_genes > 1:
            where = random.randint(0,len_genes-1)
        gene = self.get_genes()[where]
        if gene.get_direction() % 2 == 0:
            dir = random.randint(0, 1) * 2 + 1
        else:
            dir = random.randint(0, 1) * 2
        gene = Gene()
        gene.set_params(dir, 1)
        if len_genes - 1 > where > 0:
            prev_gene = self.get_genes()[where - 1]
            if prev_gene.get_direction() == dir:
                prev_gene.set_step(prev_gene.get_step() + 1)
            else:
                prev_gene.set_step(prev_gene.get_step() - 1)
                if prev_gene.get_step() == 0:
                    self.get_genes().pop(where - 1)
        else:
            self.get_genes().insert(where, gene)

        ### now we need to fix the problem
        temps = self.get_list_of_temp_points(start_point)
        end_to_fix = temps[len(temps) - 1]
        gene_to_append = Genotype.get_new_gene_to_the_end(end_to_fix, end_point)
        last_gene = self.get_genes()[len(self.get_genes())-1]
        if gene_to_append.get_direction() % 2 == last_gene.get_direction()% 2 and gene_to_append.get_direction() != last_gene.get_direction():
            last_gene.set_step(last_gene.get_step() - 1)
            if last_gene.get_step() == 0:
                self.get_genes().pop(len(self.get_genes()) - 1)
        else:
            self.get_genes().append(gene_to_append)

        self.normalize(start_point, end_point)

    def is_valid(self, start_point, end_point):
        prev_point = start_point
        for i in self.get_genes():
            prev_point = Genotype.recognize_current_point(prev_point,i.get_direction(), i.get_step())
        if prev_point == end_point:
            return True
        return False

    def no_loops(self, start_point):
        d = dict()
        tp = copy.deepcopy(self.get_genes())
        prev_point = copy.deepcopy(start_point)
        d[prev_point.to_str2()] = [0]
        for i in range(0, len(tp)):
            while tp[i].get_step() > 0:
                prev_point = Genotype.recognize_current_point(prev_point, tp[i].get_direction(), 1)
                tp[i].set_step(tp[i].get_step() - 1)
                if prev_point.to_str2() in d.keys():
                    d[prev_point.to_str2()].append(i)
                else:
                    d[prev_point.to_str2()] = [i]
        for i in d.keys():
            if len(d[i]) > 1:
                return False
        return True


    def remove_loops(self, start_point):
        d = dict()
        tp = copy.deepcopy(self.get_genes())
        prev_point = copy.deepcopy(start_point)
        beg = list()
        for i in range(0, len(tp)):
            beg.append(prev_point)
            prev_point = Genotype.recognize_current_point(prev_point, tp[i].get_direction(), tp[i].get_step())

        prev_point = copy.deepcopy(start_point)
        for i in range(0, len(tp)):
            while tp[i].get_step() > 0:
                prev_point = Genotype.recognize_current_point(prev_point, tp[i].get_direction(), 1)
                tp[i].set_step(tp[i].get_step()-1)
                if prev_point.to_str2() in d.keys():
                    d[prev_point.to_str2()].append(i)
                else:
                    d[prev_point.to_str2()] = [i]
        to_shorten = list()
        to_remove = list()
        for i in d.keys():
            if len(d[i]) > 1:
                print(d[i])
                to_shorten.append((d[i][0], i))
                for j in range(1, len(d[i])):
                    to_remove.append(d[i][j])

        for i in to_shorten:
            p = Point.from_str_to_point(i[1])
            if p.get_x()==beg[i[0]].get_x():
                diff = abs(p.get_y() - beg[i[0]].get_y())
            else:
                diff = abs(p.get_x() - beg[i[0]].get_x())
            self.get_genes()[i[0]].set_step(self.get_genes()[i[0]].get_step() - diff)
        counter = len(to_remove) - 1
        while counter >= 0:
            self.get_genes().pop(to_remove[counter])
            counter -= 1
        counter = len(self.get_genes()) - 1

        while counter >= 0:
            if self.get_genes()[counter].get_step() == 0:
                self.get_genes().pop(counter)
            counter -= 1

    def normalize(self, start_point, end_point):
        counter = 0
        while counter < len(self.get_genes()):
            if counter < len(self.get_genes()) - 1:
                if self.get_genes()[counter].get_direction() == self.get_genes()[counter + 1].get_direction():
                    self.get_genes()[counter].set_step(self.get_genes()[counter].get_step() + self.get_genes()[counter + 1].get_step())
                    self.get_genes().pop(counter + 1)
                elif self.get_genes()[counter].get_direction()% 2 == self.get_genes()[counter + 1].get_direction() % 2:
                    if self.get_genes()[counter].get_step()> self.get_genes()[counter + 1].get_step():
                        self.get_genes()[counter].set_step(self.get_genes()[counter].get_step() - self.get_genes()[counter + 1].get_step())
                        self.get_genes().pop(counter + 1)
                    elif self.get_genes()[counter].get_step()> self.get_genes()[counter + 1].get_step():
                        self.get_genes()[counter].set_step(self.get_genes()[counter + 1].get_step() - self.get_genes()[counter].get_step())
                        self.get_genes().pop(counter)
                    else:
                        self.get_genes().pop(counter + 1)
                        self.get_genes().pop(counter)
                else:
                    counter += 1
            else:
                counter += 1
        # eliminate loops
        done = False
        counter = 0
        while done is not True:
            if counter == len(self.get_genes()) - 1:
                next_point = Genotype.recognize_current_point(start_point, self.get_genes()[counter].get_direction(), self.get_genes()[counter].get_step())
                if next_point == end_point:
                    for i in range(len(self.get_genes()) - 1, counter):
                        self.get_genes().pop(i)
                    done = True
            else:
                done = True
            counter += 1

    @staticmethod
    def get_new_gene_to_the_end(end_to_fix, end_point):
        gene = Gene()
        if end_to_fix.get_x() == end_point.get_x():
            if end_to_fix.get_y() > end_point.get_y():
                gene.set_params(0,end_to_fix.get_y()-end_point.get_y())
            else:
                gene.set_params(2, end_point.get_y() - end_to_fix.get_y())
        else:
            if end_to_fix.get_x() > end_point.get_x():
                gene.set_params(3,end_to_fix.get_x()-end_point.get_x())
            else:
                gene.set_params(1, end_point.get_x() - end_to_fix.get_x())
        return gene

    def mutate(self):
        if len(self.genes) == 1:
            no_path = 0
        else:
            no_path = random.randint(0, len(self.genes)-1)

        direction = self.genes[no_path].get_direction()
        # When just straight line
        if no_path == 0 and no_path == len(self.genes)-1:
            if direction % 2 == 0:
                fgene = Gene()
                lgene = Gene()
                fdir = random.randint(0, 1)*2 + 1
                if fdir == 1:
                    ldir = 3
                else:
                    ldir = 1
                fgene.set_params(fdir,1)
                lgene.set_params(ldir,1)
                self.genes.append(lgene)
                self.genes.insert(0,fgene)
            else:
                fgene = Gene()
                lgene = Gene()
                fdir = random.randint(0, 1) * 2
                if fdir == 0:
                    ldir = 2
                else:
                    ldir = 0
                fgene.set_params(fdir, 1)
                lgene.set_params(ldir, 1)
                self.genes.append(lgene)
                self.genes.insert(0, fgene)

        # When not straight line but random segment is the first one
        elif no_path == 0:
            gene = Gene()
            if direction % 2 == 0:
                dir = random.randint(0, 1) * 2 + 1
            else:
                dir = random.randint(0, 1) * 2
            gene.set_params(dir, 1)
            if dir == self.genes[1].get_direction():
                if self.genes[1].get_step() == 1:
                    if len(self.genes) > 2:
                        if self.genes[0].get_direction() != self.genes[2].get_direction():
                            if self.genes[0].get_step() > self.genes[2].get_step():
                                self.genes[0].set_step(self.genes[0].get_step()-self.genes[2].get_step())
                                self.genes.pop(2)
                                self.genes.pop(1)
                            elif self.genes[0].get_step()>self.genes[2].get_step():
                                self.genes[2].set_step(self.genes[2].get_step() - self.genes[0].get_step())
                                self.genes.pop(1)
                                self.genes.pop(0)
                            else:
                                self.genes.pop(2)
                                self.genes.pop(1)
                                self.genes.pop(0)
                        else:
                            self.genes[0].set_step(self.genes[0].get_step()+self.genes[2].get_step())
                            self.genes.pop(2)
                            self.genes.pop(1)
                    else:
                        self.genes.pop(1)
                    self.genes.insert(0, gene)
                else:
                    self.genes[1].set_step(self.genes[1].get_step()-1)
                    self.genes.insert(0, gene)
            else:
                self.genes[1].set_step(self.genes[1].get_step() + 1)
                self.genes.insert(0, gene)

        # When not straight line but random segment is the last one
        elif no_path == len(self.genes)-1:
            last = len(self.genes)-1
            gene = Gene()
            if direction % 2 == 0:
                dir = random.randint(0, 1) * 2 + 1
            else:
                dir = random.randint(0, 1) * 2
            gene.set_params(dir, 1)
            if dir == self.genes[last-1].get_direction():
                if self.genes[last-1].get_step() == 1:
                    if len(self.genes) > 2:
                        if self.genes[last].get_direction() != self.genes[last-2].get_direction():
                            if self.genes[last].get_step() > self.genes[last-2].get_step():
                                self.genes[last].set_step(self.genes[last].get_step() - self.genes[last-2].get_step())
                                self.genes.pop(last-2)
                                self.genes.pop(len(self.genes)-2)
                            elif self.genes[last].get_step() > self.genes[last-2].get_step():
                                self.genes[last-2].set_step(self.genes[last-2].get_step() - self.genes[last].get_step())
                                self.genes.pop(last-1)
                                self.genes.pop(len(self.genes)-1)
                            else:
                                self.genes.pop(last-2)
                                self.genes.pop(len(self.genes)-2)
                                self.genes.pop(len(self.genes)-1)
                        else:
                            self.genes[last].set_step(self.genes[last].get_step() + self.genes[last-2].get_step())
                            self.genes.pop(last-2)
                            self.genes.pop(len(self.genes)-2)
                    else:
                        self.genes.pop(last - 1)
                    self.genes.append(gene)
                else:
                    self.genes[last - 1].set_step(self.genes[last - 1].get_step()-1)
                    self.genes.append(gene)
            else:
                self.genes[last - 1].set_step(self.genes[last - 1].get_step() + 1)
                self.genes.append(gene)

        # When random segment isn't last and neither it is first.
        else:
            gene = Gene()
            if direction % 2 == 0:
                dir = random.randint(0, 1) * 2 + 1
            else:
                dir = random.randint(0, 1) * 2
            gene.set_params(dir, 1)
            if dir != self.genes[no_path + 1].get_direction():
                self.genes[no_path+1].set_step(self.genes[no_path+1].get_step() + 1)
            else:
                if self.genes[no_path+1].get_step() > 1:
                    self.genes[no_path+1].set_step(self.genes[no_path+1].get_step()-1)
                else:
                    if no_path+2 == len(self.genes):
                        self.genes[no_path],self.genes[no_path+1] = self.genes[no_path + 1] , self.genes[no_path]
                    else:
                        if self.genes[no_path+2].get_direction() == self.genes[no_path].get_direction():
                            self.genes[no_path], self.genes[no_path + 1] = self.genes[no_path + 1] , self.genes[no_path]
                            self.genes[no_path + 1].set_step(self.genes[no_path + 1].get_step() + self.genes[no_path + 2].get_step())
                            self.genes.pop(no_path + 2)
                        else:
                            self.genes.pop(no_path + 2)
                            self.genes.pop(no_path)
            if dir == self.genes[no_path - 1].get_direction():
                self.genes[no_path-1].set_step(self.genes[no_path-1].get_step() + 1)
            else:
                if self.genes[no_path - 1].get_step() > 1:
                    self.genes[no_path - 1].set_step(self.genes[no_path - 1].get_step() - 1)
                else:
                    self.genes.pop(no_path - 1)

    def mutate3(self, x_dim, y_dim):
        len_genes = len(self.genes)
        where = 0
        if len_genes > 1:
            where = random.randint(0, len_genes - 1)
        gene = self.get_genes()[where]
        if gene.get_direction() % 2 == 0:
            dir = random.randint(0, 1) * 2 + 1
            step = random.randint(1, y_dim-1)
        else:
            dir = random.randint(0, 1) * 2
            step = random.randint(1, x_dim-1)
        ngene = Gene()
        ngene.set_params(dir, step)
        opp_gene = Gene()
        opp_gene.set_params(Gene.get_opp_direction(dir),step)
        mutation_segments = []
        if gene.get_step()>1 and bool(random.getrandbits(1)):
            split_point = random.randint(1, gene.get_step() - 1)
            fgene = Gene()
            lgene = Gene()
            fgene.set_params(gene.get_direction(),split_point)
            lgene.set_params(gene.get_direction(),gene.get_step()-split_point)
            if bool(random.getrandbits(1)):
                mutation_segments.append(ngene)
                mutation_segments.append(fgene)
                mutation_segments.append(opp_gene)
                mutation_segments.append(lgene)
            else:
                mutation_segments.append(fgene)
                mutation_segments.append(ngene)
                mutation_segments.append(lgene)
                mutation_segments.append(opp_gene)
        else:
            cp_gene = copy.deepcopy(gene)
            mutation_segments.append(ngene)
            mutation_segments.append(cp_gene)
            mutation_segments.append(opp_gene)

        del self.genes[where]
        self.genes[where:where] = mutation_segments
        self.normalize_v2()

    def normalize_v2(self):
        new_steps = []
        if len(self.genes) < 2:
            return
        for i in self.genes:
            if len(new_steps) == 0:
                new_steps.append(i)
                continue

            ls = new_steps[-1]
            li = len(new_steps) - 1
            if ls.get_direction() == i.get_direction():
                gene = Gene()
                gene.set_params(ls.get_direction(), (ls.get_step() + i.get_step()))
                new_steps[li]=gene
                continue

            if ls.get_direction() == Gene.get_opp_direction(i.get_direction()):
                gene = Gene()
                diff = ls.get_step() - i.get_step()
                if diff > 0:
                    ls.set_params(ls.get_direction(),diff)
                    continue
                elif diff == 0:
                    new_steps.remove(ls)
                    continue
                else:
                    ls.set_params(i.get_direction(),abs(diff))
                    continue
            new_steps.append(i)

        self.genes = new_steps



    def __str__(self):
        result = ""
        for i in self.get_genes():
            result += 'Direction: %s, step: %d\n' % (Gene.get_name_of_direction(i.get_direction()), i.get_step())
        return result


class Individual:
    def __init__(self):
        self.genotypes = list()
        self.fitness = 0
        self.pen_intersects = list()
        self.pen_path_length = 0
        self.pen_no_segments = 0
        self.pen_no_paths_outside = 0
        self.pen_path_length_outside = 0

    def get_intersects(self):
        return self.pen_intersects

    def mutation(self,data):
        if random.randint(0,10)<3:
            x = random.randint(0,len(self.genotypes)-1)
            self.genotypes[x].mutate3(data.get_x_dim(),data.get_y_dim())

    def add_genotype(self, genotype):
        self.genotypes.append(genotype)

    def get_genotypes(self):
        return self.genotypes

    def set_genotypes(self, genots):
        self.genotypes = genots

    def count_intersects(self, data):
        dct = dict()
        list_of_points = list()
        for i in range(0, len(self.get_genotypes())):
            temp = self.get_genotypes()[i].get_list_of_temp_points(data.get_connections()[i].get_start_point())
            for x in temp:
                xstr = x.to_str()
                if xstr in dct.keys():
                    dct[xstr].append(i)
                else:
                    dct[xstr] = list()
                    dct[xstr].append(i)
        for i in dct.values():
            if len(i) > 1:
                for j in i:
                    if j not in list_of_points:
                        list_of_points.append(j)
        return list_of_points

    #def get_paths_with_intersect(self):
    #    list_of_paths = list()
    #    for i in self.pen_intersects:
    #        for j in i:
    #            if j not in list_of_paths:
    #                list_of_paths.append(j)
    #    return list_of_paths

    def calculate_path_length(self):
        length = 0
        for i in self.get_genotypes():
            length += i.get_path_length()
        return length

    def count_segments(self):
        total = 0
        for i in self.get_genotypes():
            total += i.get_number_of_segments()
        return total

    def calculate_path_outside(self, data):
        total = 0
        for i in range(0, len(self.get_genotypes())):
            total += self.get_genotypes()[i].get_path_len_outside(data.get_connections()[i].get_start_point(), data.get_x_dim(), data.get_y_dim())
        return total

    def count_segments_outside(self, data):
        total = 0
        for i in range(0, len(self.get_genotypes())):
            total += self.get_genotypes()[i].get_number_of_segments_outside(data.get_connections()[i].get_start_point(), data.get_x_dim(), data.get_y_dim())
        return total

    def randomize_genotypes(self, data):
        for i in data.get_connections():
            start_point = i.get_start_point()
            end_point = i.get_end_point()
            new_genotype = Genotype()
            new_genotype.generate_genotype(start_point, end_point, data.get_x_dim(), data.get_y_dim())
            self.add_genotype(new_genotype)

    def calculate_penalty(self, data):
        self.pen_intersects = self.count_intersects(data)
        self.pen_path_length = self.calculate_path_length()
        self.pen_no_segments = self.count_segments()
        self.pen_no_paths_outside = self.count_segments_outside(data)
        self.pen_path_length_outside = self.calculate_path_outside(data)

    def calculate_adapt_function(self, weights, path_weights):
        total = 0.0
        # Mult is a variable which indicates multiplication value of each paths' weight
        if len(self.pen_intersects) > 0:
            mult = 1.0
            for i in self.pen_intersects:
                mult = float(mult) * float(path_weights[i])
            total += float(weights[0]) * float(mult)
        # total += weights[0] * self.pen_no_intersects
        total += float(weights[1]) * float(self.pen_path_length)
        total += float(weights[2]) * float(self.pen_no_segments)
        total += float(weights[3]) * float(self.pen_no_paths_outside)
        total += float(weights[4]) * float(self.pen_path_length_outside)
        return total

    def get_list_of_points_visited(self, data):
        lst = list()
        for i in range(0, len(self.get_genotypes())):
            lst.append(self.get_genotypes()[i].get_list_of_temp_points(data.get_connections()[i].get_start_point()))
        return lst

    def set_fitness(self, data, weights, path_weights):
        self.calculate_penalty(data)
        self.fitness = self.calculate_adapt_function(weights, path_weights)

    def draw_plot(self,data):
        for i in range(-2,data.get_x_dim()+2):
            for j in range(-2, data.get_y_dim()+2):
                if i<0 or i>data.get_x_dim()-1 or j<0 or j>data.get_y_dim()-1:
                    plt.plot(i, j, 'o', color='red')
                else:
                    plt.plot(i, j, 'o', color='blue')
        points = self.get_list_of_points_visited(data)
        index = 0
        for i in points:
            for j in range(0,len(i)-1):
                plt.plot([i[j].get_x(),i[j+1].get_x()] , [i[j].get_y(),i[j+1].get_y()] , COLORS[index%len(COLORS)],'-')
            index += 1

        plt.show()
    def check_if_valid(self,data):
        a = 0
        g = 0
        for i in range(0,len(self.genotypes)-1):
            a += 1
            if self.genotypes[i].is_valid(data.get_connections()[i].get_start_point(),data.get_connections()[i].get_end_point()):
                g += 1
        return (a,g)


    def mutate_genotypes(self,data):
        for i in range(0,len(self.get_genotypes())):
            cp = copy.deepcopy(self.get_genotypes()[i])
            cp.mutate2(data.get_connections()[i].get_start_point(), data.get_connections()[i].get_end_point())
            if cp.no_loops(data.get_connections()[i].get_start_point()) and cp.is_valid(data.get_connections()[i].get_start_point(), data.get_connections()[i].get_end_point()):
                self.get_genotypes()[i] = cp
        return self

    def get_fitness(self):
        return self.fitness

    def __str__(self):
        no_gent = len(self.genotypes)
        result = ""
        for i in range(0, no_gent):
            result = "Genotype " + str(i)
            result += "\n" + str(self.genotypes[i])
        return result


class Population:

    def __init__(self, data):
        self.units = list()
        self.data = data
        self.weights = list()
        self.path_weights = list()

        # Setting primal path weights
        for i in range(0, self.data.get_number_of_connections()):
            self.path_weights.append(1)
        self.weights.append(0.22)
        self.weights.append(0.04)
        self.weights.append(0.05)
        self.weights.append(3.5)
        self.weights.append(4.8)

    def mutate_population(self):
        total = 0
        good = 0
        for i in range(100):
            for j in self.units:
                j.mutation(self.data)
                (a,g) = j.check_if_valid(self.data)
                total += a
                good += g
            self.units[0].draw_plot(self.data)
            print("Total: ", total)
            print("Good: ", good)

    def calculate_fitness_and_sort(self):
        for i in self.get_units():
            i.set_fitness(self.data, self.weights, self.path_weights)
        self.units.sort(key=lambda x: x.get_fitness())
        return self.units

    def add_unit(self, individual):
        self.units.append(individual)

    def get_units(self):
        return self.units

    def roulette(self):
        total = 0
        probs = list()
        selected = list()
        self.calculate_fitness()
        for i in self.units:
            total += float(1/(i.get_fitness()))

        for i in self.units:
            probs.append(float(1/i.get_fitness())/float(total))

        for i in range(len(self.units)):
            candidate = choices(self.units, probs)[0]
            selected.append(copy.deepcopy(candidate))
        return selected

    def tournament(self, n):
        selected = list()
        positions = list()
        for i in range(len(self.units)):
            min_fit = math.inf
            x = 0
            for j in range(n):
                positions.append(random.randint(0,random.randint(0,len(self.units)-1)))
                if self.units[positions[j]].get_fitness() < min_fit:
                    min_fit = self.units[positions[j]].get_fitness()
                    x = positions[j]
            selected.append(copy.deepcopy(self.units[x]))
        return selected

    def tournament2(self):
        selected = list()
        for i in range(len(self.units)):
            a = random.randint(0,len(self.units)-1)
            b = random.randint(0,len(self.units)-1)
            while a == b:
                b = random.randint(0, len(self.units) - 1)
            if self.units[a].get_fitness()<self.units[b].get_fitness():
                selected.append(copy.deepcopy(self.units[a]))
            else:
                selected.append(copy.deepcopy(self.units[b]))
        return selected

    def test_reproduction(self):
        funit = self.get_units()[0]
        sunit = self.get_units()[1]
        funit.draw_plot(self.data)
        sunit.draw_plot(self.data)
        ind = self.reproduct(funit, sunit)
        ind.draw_plot(self.data)
        ind.mutate_genotypes()
        ind.draw_plot(self.data)
        funit.draw_plot(self.data)
        sunit.draw_plot(self.data)

    @staticmethod
    def reproduct(funit, sunit):
        paths1 = copy.deepcopy(funit.get_genotypes())
        paths2 = copy.deepcopy(sunit.get_genotypes())
        where = 1
        r = random.randint(0,10)
        if r > 8:
            return funit, sunit
        if len(paths1) > 2:
            where = random.randint(1, len(paths1)-1)
        individual = Individual()
        for i in paths1[0:where]:
            individual.add_genotype(i)
        for i in paths2[where:len(paths2)]:
            individual.add_genotype(i)

        individual2 = Individual()
        for i in paths2[0:where]:
            individual2.add_genotype(i)
        for i in paths1[where:len(paths2)]:
            individual2.add_genotype(i)

        return individual, individual2

    def test_mutation_and_print_plot(self):
        #self.units[0].draw_plot(self.data)
        temp = list()
        counter = 0
        should = 0
        lst = copy.deepcopy(self.units)
        self.units.clear()
        for i in lst:
            self.units.append(copy.deepcopy(i.mutate_genotypes(self.data)))
            for j in range(0, len(i.get_genotypes())):
                temp = i.get_genotypes()[j].get_list_of_temp_points(self.data.get_connections()[j].get_start_point())
                if temp[len(temp)-1] == self.data.get_connections()[j].get_end_point():
                    counter += 1
                should += 1
                temp.clear()
        #self.units[0].draw_plot(self.data)
        print("Was ok: %d" % counter)
        print("Should be ok: %d" % should)

    def generate_new_population(self, no_units):
        for i in range(0, no_units):
            new_individual = Individual()
            new_individual.randomize_genotypes(self.data)
            self.add_unit(new_individual)

    def show_population(self):
        counter = 1
        for i in self.get_units():
            print("Unit %d:\n" % counter)
            count2 = 1
            for j in i.get_genotypes():
                print("Genotype %d\n " % count2)
                print(str(j) + "\n")
                count2 += 1
            counter += 1

    def calculate_fitness(self):
        for i in self.units:
            i.set_fitness(self.data, self.weights, self.path_weights)

    def test_adapt_function(self):
        for i in range(0, len(self.units)):
            self.units[i].calculate_penalty(self.data)
            print("Unit %d adapt function: " % (i+1))
            print(self.units[i].calculate_adapt_function(self.weights, self.path_weights))

    def test_population(self):
        counter = 1
        for i in self.get_units():
            print("Individual " + str(counter) + "\n")
            print("    Intersects: " + str(i.count_intersects(self.data)) + "\n")
            print("    Segments: " + str(i.count_segments()) + "\n")
            print("    Path length: " + str(i.calculate_path_length()) + "\n")
            print("    Path length outside: " + str(i.calculate_path_outside(self.data)) + "\n")
            print("    Segments outside: " + str(i.count_segments_outside(self.data)) + "\n")
            counter += 1

    def simulate(self, n):
        last_fitness = math.inf
        for i in range(n):
            self.calculate_fitness_and_sort()
            best = self.units[0]
            if last_fitness <= best.get_fitness():
                paths = self.units[0].get_intersects()
                for j in paths:
                    self.path_weights[j] = self.path_weights[j] + 1
            last_fitness = best.get_fitness()

            #print("Population: %d, best fitness:" % i)
            #str_total = str(best.get_fitness())
            #x = str_total.replace(".", ",")
            #print(x)
            counter = 0
            lst = [best]
            #selected = self.tournament(tur)
            selected = self.roulette()
            while counter < len(selected) - 1:
                sol = Population.reproduct(selected[counter], selected[counter + 1])
                lst.append(sol[0])
                lst.append(sol[1])
                counter += 2
            if len(lst) > len(self.units):
                lst.pop()
            self.units.clear()
            for j in lst:
                self.units.append(j)
            for j in self.units:
                j.mutation(self.data)

        # The end show results:
        self.calculate_fitness_and_sort()
        best = self.units[0]
        best.draw_plot(self.data)
        #print(best.get_intersects())
        #print(best.pen_path_length)
        #print(best.pen_no_segments)
        #print(best.pen_no_paths_outside)
        #print(best.pen_path_length_outside)
        #print("Total: ")
        print(Population.get_uniform_fitness(best))

    @staticmethod
    def get_uniform_fitness(best):
        total = 1.0
        total = total + float(50.0 * len(best.get_intersects()))
        total = total + float(0.8 * best.pen_path_length)
        total = total + float(1.2 * best.pen_no_segments)
        total = total + float(5 * best.pen_no_paths_outside)
        total = total + float(2 * best.pen_path_length_outside)
        str_total = str(total)
        x = str_total.replace(".", ",")
        return x

    def simulation(self, n):
        for i in range(n):

            # Firstly calculate fitness function for entire population
            fit = list()
            for j in self.units:
                j.set_fitness(self.data, self.weights, self.path_weights)
                fit.append(j.get_fitness())

            # Then check which individual has the smallest fitness and add: 1 to every path that has any intersection
            min_index = fit.index(min(fit))
            paths = self.units[min_index].get_paths_with_intersect()
            for j in paths:
                self.path_weights[j] = self.path_weights[j] + 1
            print(self.path_weights)

            while counter < len(selected) - 1:
                lst.append(Population.reproduct(selected[counter], selected[counter+1]))
                lst.append(Population.reproduct(selected[counter+1], selected[counter]))
                counter += 2

            # Select parents with one of the selection algorithm
            selected = self.roulette()
            counter = 0
            lst = list()
            while counter < len(selected) - 1:
                lst.append(Population.reproduct(selected[counter], selected[counter+1]))
                lst.append(Population.reproduct(selected[counter+1], selected[counter]))
                counter += 2
            if len(selected) % 2 == 1:
                lst.append(copy.deepcopy(self.units[min_index]))
            self.units.clear()
            for x in lst:
                self.units.append(x)

            # After selection and reproduction -> mutate
            for j in self.units:
                j.mutate_genotypes(self.data)

        self.units[0].set_fitness(self.data, self.weights, self.path_weights)
        min_fit = self.units[0].get_fitness()
        for j in self.units:
            j.set_fitness(self.data, self.weights, self.path_weights)
            if min_fit>=j.get_fitness():
                j.draw_plot(self.data)
                print(j.pen_intersects)
                #print(j.count_intersects(self.data))
                min_fit = j.get_fitness()

    def test_mutation(self):
        self.units[0].draw_plot(self.data)
        self.units[0].mutation(self.data)
        self.units[0].draw_plot(self.data)

    def test_reproduct(self):
        self.units[0].draw_plot(self.data)
        self.units[1].draw_plot(self.data)
        a, b = self.reproduct(self.units[0],self.units[1])
        a.draw_plot(self.data)
        b.draw_plot(self.data)
    def __str__(self):
        no_units = len(self.units)
        result = "Population\n"
        for i in range(0, no_units):
            result += "Unit " + str(i+1)
            result += "\n" + str(self.units[i])
        return result
