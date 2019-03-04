import numpy as np
import fetchmaker
from scipy.stats import binom_test, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# get rotweiler taillength data and calculate mean and standard deviation
rottweiler_tl1 = fetchmaker.get_tail_length("rottweiler")
print("Rotweiler taillength mean is {}".format(np.mean(rottweiler_tl1)))
print("Rotweiler taillength sd is {}".format(np.std(rottweiler_tl1)))


# Test is whippet dogs are more likely to be rescued against the population mean rescue rate of 8%

# Get every whippet dog with attribute "is rescue". In this database that is every whippet dog
whippet_rescue = fetchmaker.get_is_rescue("whippet")
# Get subset of whippet_rescue with value of 1. These are the rescued whippet dogs
num_whippet_rescues = np.count_nonzero(whippet_rescue)
# Get all whippet dogs
num_whippets = np.size(whippet_rescue)
# a binomial test to test the null hypotheses that there is no difference in rescue rates
binom_rescue = binom_test(num_whippet_rescues, num_whippets, .08)
print("\nP-value: {}".format(binom_rescue))
if binom_rescue < .05:
    print("There is a significant difference in rescue rates between whippet dogs and the entire population")
if binom_rescue >= 0.05:
    print("There is no significant difference in rescue rates between the population and whippet dogs")


# Test if there is a difference between whippet, terrier, and pitbull weights

# Get all weights
whippet_weight = fetchmaker.get_weight("whippet")
terrier_weight = fetchmaker.get_weight("terrier")
pitbull_weight = fetchmaker.get_weight("pitbull")
# Use ANOVA test to test if there is a difference between the three breeds
anova_test = f_oneway(whippet_weight, terrier_weight, pitbull_weight)
print("\nP-value: {}".format(anova_test.pvalue))
if anova_test.pvalue < .05:
    print("There is a weigt difference between whippets, terriers, and pitbulls")
if anova_test.pvalue >= .05:
    print("There is no significant difference between the weights of whippets, terriers, and pitbulls")

# pairwise Tukey HSD
# put all values in one list
values = np.concatenate([whippet_weight, terrier_weight, pitbull_weight])
# create labels with correct length
labels = ['Whippet'] * len(whippet_weight) + ['Terrier'] * len(terrier_weight) + ['Pitbull'] * len(pitbull_weight)
outliers = pairwise_tukeyhsd(values, labels, 0.05)
print("\n{}".format(outliers))
print("Conclusion: Pitbulls and Whippet have similar weight while other combinations are significantly different")

# Test if there is a significant  color difference between poodles and shih tzus
poodle_colors = fetchmaker.get_color("poodle")
shihtzu_colors = fetchmaker.get_color("shihtzu")

black_p = np.count_nonzero(poodle_colors == "black")
black_s = np.count_nonzero(shihtzu_colors == "black")
brown_p = np.count_nonzero(poodle_colors == "brown")
brown_s = np.count_nonzero(shihtzu_colors == "brown")
gold_p = np.count_nonzero(poodle_colors == "gold")
gold_s = np.count_nonzero(shihtzu_colors == "gold")
grey_p = np.count_nonzero(poodle_colors == "grey")
grey_s = np.count_nonzero(shihtzu_colors == "grey")
white_p = np.count_nonzero(poodle_colors == "white")
white_s = np.count_nonzero(shihtzu_colors == "white")

np_color_table = np.array([["# black poodles", "# black shih tzus"],
                           ["# brown poodles", "# brown shih tzus"],
                           ["# gold poodles", "# gold shih tzus"],
                           ["# grey poodles", "# grey shih tzus"],
                           ["# white poodles", "# white shih tzus"]])
print("\ncolor table layout:\n", np_color_table)
colors = [[black_p, black_s], [brown_p, brown_s], [gold_p, gold_s], [grey_p, grey_s], [white_p, white_s]]
print("color table values:", colors)

_, color_pval, _, _ = chi2_contingency(colors)
print("\nThe p-value of the color difference between poodles and shihtzu's is {}".format(color_pval))
if color_pval < 0.05:
    print("There is a significant difference between poodle and shihtzu colors")
if color_pval >= 0.05:
    print("There is no significant difference between poodle and shitzu colors")
