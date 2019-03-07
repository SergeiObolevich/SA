import numpy as np
from functools import reduce
from scipy.stats.distributions import chi2
from scipy.stats import t


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


def expected_value(variable):
    return reduce((lambda x, y: x + y), variable) / len(variable)


def variance(variable):
    return expected_value(np.square(variable)) - pow(expected_value(variable), 2)


def displaced_variance(variable):
    n = len(variable)
    return variance(variable) * n / (n - 1)


def confidence_interval(variable, alpha):
    coefficient = np.abs(t.ppf(alpha / 2, df=(len(variable) - 1)))
    return coefficient * np.sqrt(displaced_variance(variable) / len(variable))


def interval_border(variable, alpha):
    length = len(variable)
    high = chi2.ppf(alpha / 2, df=(length - 1))
    low = chi2.ppf(1 - alpha / 2, df=(length - 1))
    variable_displaced_variance = displaced_variance(variable)
    low_border = variable_displaced_variance * (length - 1) / low
    high_border = variable_displaced_variance * (length - 1) / high
    return low_border, high_border


def check_hypothesis_with_variance(first_expected, second_expected, first_variance,
                                   second_variance, first_amount, second_amount):
    return np.abs(first_expected - second_expected) / np.sqrt((first_variance / len(first_amount))
                                                              + (second_variance / len(second_amount)))


def check_hypothesis_without_variance(first_expected, second_expected, first_displaced_variance,
                                      second_displaced_variance, first_amount, second_amount):
    return np.abs(first_expected - second_expected) / np.sqrt(
            (len(first_amount) - 1) * first_displaced_variance + (len(second_amount) - 1) * second_displaced_variance) \
           * np.sqrt(((len(first_amount) * len(second_amount)) * (len(first_amount) + len(second_amount) - 2)) / (
                   len(first_amount) + len(second_amount)))


def main():
    rectal_temperature, total_protein, parsed_data_size = file_parsing('13-horse-colic.txt', 3, 19)
    rectal_temperature = list(map(float, rectal_temperature))
    total_protein = list(map(float, total_protein))

    alpha = 0.05

    expected_value_rectal_temperature = expected_value(rectal_temperature)
    expected_value_total_protein = expected_value(total_protein)
    rectal_temperature_variance = variance(rectal_temperature)
    total_protein_variance = variance(total_protein)
    rectal_temperature_displaced_variance = displaced_variance(rectal_temperature)
    total_protein_displaced_variance = displaced_variance(total_protein)
    rectal_temperature_standard_deviation = np.sqrt(variance(rectal_temperature))
    total_protein_standard_deviation = np.sqrt(variance(total_protein))

    print('Expected value of average rectal temperature = ', expected_value_rectal_temperature)
    print('Expected value of average total protein = ', expected_value_total_protein)
    print('Variance value of average rectal temperature = ', rectal_temperature_variance)
    print('Variance value of average total protein = ', total_protein_variance)
    print('Displaced variance value of average rectal temperature = ', rectal_temperature_displaced_variance)
    print('Displaced variance value of average total protein = ', total_protein_displaced_variance)
    print('Standard deviation value of average rectal temperature = ', rectal_temperature_standard_deviation)
    print('Standard deviation value of average total protein = ', total_protein_standard_deviation)

    print('Confidence interval of average rectal temperature expected value: {} < E(x) < {}'.format(
        expected_value_rectal_temperature - confidence_interval(rectal_temperature, alpha),
        expected_value_rectal_temperature + confidence_interval(rectal_temperature, alpha)
    ))

    print('Confidence interval of average total protein expected value: {} < E(x) < {}'.format(
        expected_value_total_protein - confidence_interval(total_protein, alpha),
        expected_value_total_protein + confidence_interval(total_protein, alpha)
    ))

    temperature_low_border, temperature_high_border = interval_border(rectal_temperature, alpha)
    print('Confidence interval of average temperature variance: {} < sigma^2 < {}'.format(
        temperature_low_border,
        temperature_high_border
    ))

    protein_low_border, protein_high_border = interval_border(total_protein, alpha)
    print('Confidence interval of average protein variance: {} < sigma^2 < {}'.format(
        protein_low_border,
        protein_high_border
    ))

    print('Check hypothesis with know variance: ', check_hypothesis_with_variance(
        expected_value_rectal_temperature, expected_value_total_protein, rectal_temperature_variance,
        total_protein_variance, rectal_temperature, total_protein))
    print('Check hypothesis with unknown variance: ', check_hypothesis_without_variance(
        expected_value_rectal_temperature, expected_value_total_protein, rectal_temperature_displaced_variance,
        total_protein_displaced_variance, rectal_temperature, total_protein))


if __name__ == '__main__':
    main()