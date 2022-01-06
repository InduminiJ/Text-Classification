#David Neudorf, 101029913
#COMP4106
#2021-03-14

import numpy as np
import enum

class breed_indices(enum.Enum):
    beagle=0
    corgi=1
    husky=2
    poodle=3

class characteristics(enum.Enum):
    girth  = [[41,6],[53,9],[66,10],[61,9]]
    height = [[37,4],[27,3],[55,6],[52,7]]
    weight = [[10,2],[12,2],[22,6],[26,8]]
    pr     = [0.3,0.21,0.14,0.35]
    #pr     = [0.25,0.25,0.25,0.25]

def P_char_given_breed(char, breed, x):
    index = breed_indices[breed].value
    char_arr = characteristics[char].value

    mu    = char_arr[index][0]
    sigma = char_arr[index][1]

    prob = np.exp((-1/2)*pow(((x-mu)/sigma),2))/np.sqrt(2*np.pi*pow(sigma,2))
    return prob

def naive_bayes_classifier(input):
    # input is a three element list with [girth, height, weight] #cm cm kg

    #most_likely_class is a string indicating the most likely class, either "beagle", "corgi", "husky", or "poodle"
    # class_probabilities is a four element list indicating the probability of each class in the order [beagle probability, corgi probability, husky probability, poodle probability]
    argmax=0
    most_likely_class=0
    class_probabilities=[]
    pr_evidence = 0
    for i in breed_indices:
        b = i.name
        val = P_char_given_breed("girth",b,input[0]) * P_char_given_breed("height",b,input[1]) * P_char_given_breed("weight",b,input[2]) * characteristics["pr"].value[i.value]
        pr_evidence+=val
        class_probabilities.append(val)
        if(argmax < val):
            argmax=val
            most_likely_class=b

    class_probabilities = [i/pr_evidence for i in class_probabilities]
    return most_likely_class, class_probabilities


class fuzzy_data(enum.Enum):
    girth  = [[0,0,40,50],[40,50,60,70],[60,70,100,100]]
    height = [[0,0,25,40],[25,40,50,60],[50,60,100,100]]
    weight = [[0,0,5,15], [5,15,20,40], [20,40,100,100]]


def trapezoidal(x,char):
    param_sets = fuzzy_data[char].value
    out = []
    for p in param_sets:
        if(x<=p[0] or x>=p[3]):
            out.append(0)
        elif(p[0]<x<p[1]):
            out.append((x-p[0])/(p[1]-p[0]))
        elif(p[1]<=x<=p[2]):
            out.append(1)
        elif(p[2]<x<p[3]):
            out.append((p[3]-x)/(p[3]-p[2]))
    return out

#Goguen t-norm and s-norm
def t(x,y):
    return x*y
def s(x,y):
    return x+y - x*y

def fuzzy_classifier(input):
    # input is a three element list with [girth, height, weight]
    # highest_membership_class is a string indicating the highest membership class, either "beagle", "corgi", "husky", or "poodle"
    # class_memberships is a four element list indicating the membership in each class in the order [beagle probability, corgi probability, husky probability, poodle probability]

    girth_data = trapezoidal(input[0],"girth")
    height_data = trapezoidal(input[1],"height")
    weight_data = trapezoidal(input[2],"weight")

    height_med   = height_data[1]
    girth_small  = girth_data[0]
    weight_small = weight_data[0]
    rule_str_1 = t(height_med, s(girth_small, weight_small))

    girth_med    = girth_data[1]
    height_short = height_data[0]
    weight_med   = weight_data[1]
    rule_str_2 = t(girth_med, t(height_short, weight_med))

    girth_large = girth_data[2]
    height_tall = height_data[2]
    rule_str_3 = t(girth_large, t(height_tall, weight_med))

    weight_large = weight_data[2]
    rule_str_4 = t(s(girth_med, height_med), weight_large)

    class_memberships = [rule_str_1, rule_str_2, rule_str_3, rule_str_4]
    highest_membership_class = breed_indices(class_memberships.index(max(class_memberships))).name


    return highest_membership_class, class_memberships


if __name__ == '__main__':
    x=[99,99,99]
    print(naive_bayes_classifier(x))
    print(fuzzy_classifier(x))
