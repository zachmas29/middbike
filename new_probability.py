import constants as constants
import numpy as np

def calculate_probability(time, source, destination):
    prob = 0
    if 21 <= time or time < 7:
        prob = constants.ninepm_to_sevenam[source][destination]
    if 7 <= time < 8:
        prob = constants.sevenam_to_eightam[source][destination]
    if 8 <= time < 12:
        prob = constants.eightam_to_twelvepm[source][destination]
    if 12 <= time < 16:
        prob = constants.twelvepm_to_fourpm[source][destination]
    if 16 <= time < 21:
        prob = constants.fourpm_to_ninepm[source][destination]
    return prob