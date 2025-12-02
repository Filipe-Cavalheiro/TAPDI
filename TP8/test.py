import numpy as np

Image = [[1, 23, 46, 72],
         [42, 60, 42, 6],
         [43, 95, 97, 44],
         [88, 78, 23, 31]]

Template = [[62, 10, 20, 16], 
            [51, 92, 80, 11], 
            [89, 52, 52, 80], 
            [99, 67, 9, 59]]

image_sum = [sum(x) for x in Image]
Template_sum = [sum(x) for x in Template]

total_image = sum(image_sum)
total_template = sum(Template_sum)


wh_ = 1/(len(Image)*len(Image[0]))
total = 0

for i in range(len(Image)):
    for j in range(len(Image[0])):
        T = Template[j][i] - total_template*wh_
        I = Image[j][i] - total_image*wh_
        total += T*I

total_image_norm = 0
total_template_norm = 0

for i in range(len(Image)):
    for j in range(len(Image[0])):
        total_image_norm = Image[j][i]**2
        total_template_norm += Template[j][i]**2

total = total/(np.sqrt(total_image_norm * total_template_norm))
print(np.round(total, 2))