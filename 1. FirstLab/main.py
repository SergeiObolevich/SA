import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter


def file_parsing(file_name, first_attribute_number, second_attribute_number):
    first_attributes = []
    second_attributes = []
    with open(file_name, 'r') as file:
        for line in file:
            parsed_data = line.split()
            if parsed_data[first_attribute_number] != '?':
                if parsed_data[second_attribute_number] != '?':
                    first_attributes.append(float(parsed_data[first_attribute_number]))
                    second_attributes.append(float(parsed_data[second_attribute_number]))
    return first_attributes, second_attributes, len(first_attributes)


def normalize_variable(variable, size):
    for key, value in variable.items():
        variable[key] = value / size
    return variable


def expected_value(variable):
    variable_map = normalize_variable(Counter(variable), len(variable))
    variable_keys = list(variable_map.keys())
    variable_values = list(variable_map.values())
    return np.average(variable_keys, weights=variable_values)


def variance(variable):
    return expected_value(np.square(variable)) - pow(expected_value(variable), 2)


def main():
    rectal_temperature, total_protein, parsed_data_size = file_parsing('13-horse-colic.txt', 3, 19)
    rectal_temperature = list(map(float, rectal_temperature))
    total_protein = list(map(float, total_protein))
    rectal_temperature_vertical = np.array(list(zip(rectal_temperature)))
    total_protein_vertical = np.array(list(zip(total_protein)))

    regression = LinearRegression()
    regression.fit(rectal_temperature_vertical, total_protein_vertical)
    rectal_temperature_predict = regression.predict(rectal_temperature_vertical)

    print('Expected value of average rectal temperature = ', expected_value(rectal_temperature))
    print('Expected value of average total protein = ', expected_value(total_protein))
    print('Variance value of average rectal temperature = ', variance(rectal_temperature))
    print('Variance value of average total protein = ', variance(total_protein))
    print('Standard deviation value of average rectal temperature = ', np.sqrt(variance(rectal_temperature)))
    print('Standard deviation value of average total protein = ', np.sqrt(variance(total_protein)))
    print('Correlation coefficient value = ', np.corrcoef(rectal_temperature, total_protein)[0, 1])

    figure = plt.figure(num=None, figsize=(6, 5), dpi=100, facecolor='w', edgecolor='w')
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlabel('Average value of rectal temperature')
    ax.set_ylabel('Average value of total protein')
    plt.plot(rectal_temperature, total_protein, color='k', marker='.', linestyle='none', markersize=2)
    plt.plot(rectal_temperature_vertical, rectal_temperature_predict, color='y', marker=',', linestyle='-')
    plt.show()


if __name__ == '__main__':
    main()