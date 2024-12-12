# -*- coding: utf-8 -*-
"""LR_2_Task_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hy3CXYRgE8KUMy20PJ1D2l31UcIRE5j9
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
temp_change = ctrl.Antecedent(np.arange(-10, 11, 1), 'temp_change')
ac_control = ctrl.Consequent(np.arange(-90, 91, 1), 'ac_control')

temperature['very_cold'] = fuzz.trimf(temperature.universe, [0, 0, 10])
temperature['cold'] = fuzz.trimf(temperature.universe, [5, 15, 25])
temperature['normal'] = fuzz.trimf(temperature.universe, [20, 25, 30])
temperature['warm'] = fuzz.trimf(temperature.universe, [25, 35, 45])
temperature['very_warm'] = fuzz.trimf(temperature.universe, [40, 50, 50])

temp_change['negative'] = fuzz.trimf(temp_change.universe, [-10, -10, 0])
temp_change['zero'] = fuzz.trimf(temp_change.universe, [-1, 0, 1])
temp_change['positive'] = fuzz.trimf(temp_change.universe, [0, 10, 10])

ac_control['big_left'] = fuzz.trimf(ac_control.universe, [-90, -90, -45])
ac_control['small_left'] = fuzz.trimf(ac_control.universe, [-45, -15, 0])
ac_control['neutral'] = fuzz.trimf(ac_control.universe, [-5, 0, 5])
ac_control['small_right'] = fuzz.trimf(ac_control.universe, [0, 15, 45])
ac_control['big_right'] = fuzz.trimf(ac_control.universe, [45, 90, 90])

rule1 = ctrl.Rule(temperature['very_warm'] & temp_change['positive'], ac_control['big_left'])
rule2 = ctrl.Rule(temperature['very_warm'] & temp_change['negative'], ac_control['small_left'])
rule3 = ctrl.Rule(temperature['warm'] & temp_change['positive'], ac_control['big_left'])
rule4 = ctrl.Rule(temperature['warm'] & temp_change['negative'], ac_control['neutral'])
rule5 = ctrl.Rule(temperature['very_cold'] & temp_change['negative'], ac_control['big_right'])
rule6 = ctrl.Rule(temperature['very_cold'] & temp_change['positive'], ac_control['small_right'])
rule7 = ctrl.Rule(temperature['cold'] & temp_change['negative'], ac_control['big_right'])
rule8 = ctrl.Rule(temperature['cold'] & temp_change['positive'], ac_control['neutral'])
rule9 = ctrl.Rule(temperature['very_warm'] & temp_change['zero'], ac_control['big_left'])
rule10 = ctrl.Rule(temperature['warm'] & temp_change['zero'], ac_control['small_left'])
rule11 = ctrl.Rule(temperature['very_cold'] & temp_change['zero'], ac_control['big_right'])
rule12 = ctrl.Rule(temperature['cold'] & temp_change['zero'], ac_control['small_right'])
rule13 = ctrl.Rule(temperature['normal'] & temp_change['positive'], ac_control['small_left'])
rule14 = ctrl.Rule(temperature['normal'] & temp_change['negative'], ac_control['small_right'])
rule15 = ctrl.Rule(temperature['normal'] & temp_change['zero'], ac_control['neutral'])

ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
ac_sim = ctrl.ControlSystemSimulation(ac_ctrl)

ac_sim.input['temperature'] = 28
ac_sim.input['temp_change'] = 2

ac_sim.compute()

print(f"{ac_sim.output['ac_control']:.2f}")

ac_control.view(sim=ac_sim)