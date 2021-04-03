from data_import import Data
from population import Population

if __name__ == '__main__':
    new_data = Data()
    new_data.set_data_from_filename('zad2.txt')
    print(str(new_data))
    population = Population(new_data)
    population.generate_new_population(100)
    #population.simulate(200,20)
    x = 67
    #while x<150:
    #    for i in range(10):
    #        population.generate_new_population(20)
    #        population.simulate(2)
    #    print("===============")
    #    x+=30
    for i in range(1):
        population.generate_new_population(500)
        for j in population.get_units():
            population.calculate_fitness_and_sort()
            print(Population.get_uniform_fitness(j))
    #population.test_mutation()
    #population.test_reproduct()
    #population.mutate_population()
    #population.show_population()
    #population.test_population()
    #population.test_adapt_function()
    #population.test_mutation_and_print_plot()
    #population.test_reproduction()
    #population.tournament()

