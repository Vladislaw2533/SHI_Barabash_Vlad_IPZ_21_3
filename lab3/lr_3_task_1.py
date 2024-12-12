# -*- coding: utf-8 -*-
"""LR_2_Task_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hy3CXYRgE8KUMy20PJ1D2l31UcIRE5j9
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
pressure = ctrl.Antecedent(np.arange(0, 101, 1), 'pressure')

hot_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'hot_valve')     # Кран гарячої води
cold_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'cold_valve')   # Кран холодної води

temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['cool'] = fuzz.trimf(temperature.universe, [0, 50, 75])
temperature['warm'] = fuzz.trimf(temperature.universe, [50, 75, 100])
temperature['hot'] = fuzz.trimf(temperature.universe, [75, 100, 100])

pressure['low'] = fuzz.trimf(pressure.universe, [0, 0, 50])
pressure['medium'] = fuzz.trimf(pressure.universe, [25, 50, 75])
pressure['high'] = fuzz.trimf(pressure.universe, [50, 100, 100])

hot_valve['big_left'] = fuzz.trimf(hot_valve.universe, [-90, -90, -45])
hot_valve['medium_left'] = fuzz.trimf(hot_valve.universe, [-60, -30, 0])
hot_valve['small_left'] = fuzz.trimf(hot_valve.universe, [-30, 0, 30])
hot_valve['neutral'] = fuzz.trimf(hot_valve.universe, [-10, 0, 10])
hot_valve['small_right'] = fuzz.trimf(hot_valve.universe, [0, 30, 60])
hot_valve['medium_right'] = fuzz.trimf(hot_valve.universe, [30, 60, 90])
hot_valve['big_right'] = fuzz.trimf(hot_valve.universe, [45, 90, 90])

cold_valve['big_left'] = fuzz.trimf(cold_valve.universe, [-90, -90, -45])
cold_valve['medium_left'] = fuzz.trimf(cold_valve.universe, [-60, -30, 0])
cold_valve['small_left'] = fuzz.trimf(cold_valve.universe, [-30, 0, 30])
cold_valve['neutral'] = fuzz.trimf(cold_valve.universe, [-10, 0, 10])
cold_valve['small_right'] = fuzz.trimf(cold_valve.universe, [0, 30, 60])
cold_valve['medium_right'] = fuzz.trimf(cold_valve.universe, [30, 60, 90])
cold_valve['big_right'] = fuzz.trimf(cold_valve.universe, [45, 90, 90])

rule1 = ctrl.Rule(temperature['hot'] & pressure['high'],
                  (hot_valve['medium_left'], cold_valve['medium_right']))
rule2 = ctrl.Rule(temperature['hot'] & pressure['medium'],
                  cold_valve['medium_right'])
rule3 = ctrl.Rule(temperature['warm'] & pressure['high'],
                  hot_valve['small_left'])
rule4 = ctrl.Rule(temperature['warm'] & pressure['low'],
                  (hot_valve['small_right'], cold_valve['small_right']))
rule5 = ctrl.Rule(temperature['warm'] & pressure['medium'],
                  hot_valve['neutral'])
rule6 = ctrl.Rule(temperature['cool'] & pressure['high'],
                  (hot_valve['medium_right'], cold_valve['medium_left']))
rule7 = ctrl.Rule(temperature['cool'] & pressure['medium'],
                  (hot_valve['medium_right'], cold_valve['small_left']))
rule8 = ctrl.Rule(temperature['cold'] & pressure['low'],
                  hot_valve['big_right'])
rule9 = ctrl.Rule(temperature['cold'] & pressure['high'],
                  (hot_valve['medium_left'], cold_valve['medium_right']))
rule10 = ctrl.Rule(temperature['warm'] & pressure['high'],
                   (hot_valve['small_left'], cold_valve['small_left']))
rule11 = ctrl.Rule(temperature['warm'] & pressure['low'],
                   (hot_valve['small_right'], cold_valve['small_right']))

valve_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6,
                                 rule7, rule8, rule9, rule10, rule11])
valve_sim = ctrl.ControlSystemSimulation(valve_ctrl)

valve_sim.input['temperature'] = 70
valve_sim.input['pressure'] = 50

valve_sim.compute()

print(f"Кут для гарячого крану: {valve_sim.output['hot_valve']:.2f}")
print(f"Кут для холодного крану: {valve_sim.output['cold_valve']:.2f}")

hot_valve.view(sim=valve_sim)
cold_valve.view(sim=valve_sim)