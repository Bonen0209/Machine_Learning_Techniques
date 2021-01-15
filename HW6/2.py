for i in range(1, 50):
    sum_param = 0
    
    sum_param += 20 * (i - 1)
    sum_param += i * (50 - 1 - i)
    sum_param += (50 - i) * 3
    
    sum_params.append((sum_param, i, (50 - i)))

for i in range(1, 50):
    for j in range(1, 50-i):
        sum_param = 0
        
        sum_param += 20 * (i - 1)
        sum_param += i * (j - 1)
        sum_param += j * (50 - 1 - i - j)
        sum_param += (50 - i - j) * 3
        
        sum_params.append((sum_param, i, j, (50 - i - j)))

for i in range(1, 50):
    for j in range(1, 50-i):
        for k in range(1, 50-i-j):
            sum_param = 0
            
            sum_param += 20 * (i - 1)
            sum_param += i * (j - 1)
            sum_param += j * (k - 1)
            sum_param += k * (50 - 1 - i - j - k)
            sum_param += (50 - i - j - k) * 3
            
            sum_params.append((sum_param, i, j, k, (50 - i - j - k)))

print(max(sum_params))
