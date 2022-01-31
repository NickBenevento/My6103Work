#%%
# print("Hello world!")


#%%
# Question 1: Create a Markdown cell with the followings:
# Two paragraphs about yourself. In one of the paragraphs, give a hyperlink of a website 
# that you want us to see. Can be about yourself, or something you like.

# %%[markdown]
# I am a graduate student in the Computer Science masters program at GW. I currently work for a company named [Blackcape](blackcape.io), who develops mission 
# applications applying data analytics and machine learning for government and commercial clients.
#
# In my free time, I enjoy doing almost anything outside like hiking, rock climbing, and snowboarding (when I'm back in my home town in Vermont). When not outside, I like
# to watch movies and play video games with my friends.

#%%
# Question 2: Create
# a list of all the class titles that you are planning to take in the data science program. 
# Have at least 6 classes, even if you are not a DS major
# Then print out the last entry in your list.
classes = ['Intro to Data Mining', 'Machine Learning I: Algorithm Analysis', 'Machine Learning II: Data Analysis', 'Natural Language Processing for Data Science', 'Visualization of Complex Data', 'Cloud Computing']
print(classes[-1])


#%%
# Question 3: After you completed question 2, you feel Intro to data mining is too stupid, so you are going 
# to replace it with Intro to Coal mining. Do that in python here.
classes[classes.index('Intro to Data Mining')] = 'Intro to Coal Mining'
print(classes)


#%%
# Question 4: Before you go see your acadmic advisor, you are 
# asked to create a python dictionary of the classes you plan to take, 
# with the course number as key. Please do that. Don't forget that your advisor 
# probably doesn't like coal. And that coal mining class doesn't even have a 
# course number.
course_keys = { 37357: 'Intro to Data Mining',
                33773: classes[1],
                33544: classes[2],
                35572: classes[3],
                33328: classes[4],
                34512: classes[5]}
print(course_keys)

#%%
# Question 5: print out and show your advisor how many 
# classes (print out the number, not the list/dictionary) you plan 
# to take.
[print(key) for key in course_keys.keys()]

#%%
# Question 6: Using loops 
# Goal: print out the list of days (31) in Jan 2021 like this
# Sat - 2022/1/1
# Sun - 2022/1/2
# Mon - 2022/1/3
# Tue - 2022/1/4
# Wed - 2022/1/5
# Thu - 2022/1/6
# Fri - 2022/1/7
# Sat - 2022/1/8
# Sun - 2022/1/9
# Mon - 2022/1/10
# Tue - 2022/1/11
# Wed - 2022/1/12
# Thu - 2022/1/13
# ...
# You might find something like this useful, especially if you use the remainder property x%7
dayofweektuple = ('Sun','Mon','Tue','Wed','Thu','Fri','Sat') # day-of-week-tuple
for i in range(31):
    print(f'{dayofweektuple[(i-1)%7]} - 2022/1/{i+1}')


# %%[markdown]
# # Additional Exercise: 
# Choose three of the five exercises below to complete.
#%%
# =================================================================
# Class_Ex1: 
# Write python codes that converts seconds, say 257364 seconds,  to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------



#%%
# =================================================================
# Class_Ex2: 
# Write a python codes to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# Hint: one way is to create three nested loops.
# ----------------------------------------------------------------
letters = ['A', 'B', 'C']
for a in letters:
    for b in letters:
        for c in letters:
            if a != b and a != c and b != c:
                print(a + b + c)



#%%
# =================================================================
# Class_Ex3: 
# Write a python codes to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------





#%%
# =================================================================
# Class_Ex4: 
# Suppose we wish to draw a triangular tree, and its height is provided 
# by the user, like this, for a height of 5:
#      *
#     ***
#    *****
#   *******
#  *********
# ----------------------------------------------------------------
height = int(input('Input the height of the tree: '))
for i in range(height+1):
    [print(' ', end='') for j in range(height-i)]
    [print('*', end='') for j in range(2*i+1)]
    print()

print()


#%%
# =================================================================
# Class_Ex5: 
# Write python codes to print prime numbers up to a specified 
# values, say up to 200.
# ----------------------------------------------------------------
import math

num = int(input('Enter a number to print prime numbers: '))
for i in range(2, num):
    prime = True

    # Only need to check factors up until the sqrt of the num
    for j in range(2, int(math.sqrt(i)+1)):
        if i % j == 0:
            prime = False
            break

    if prime:
        print(i)




# =================================================================
