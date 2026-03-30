# ECE4723 Senior Design - Invisivisor

### How to SCP a file onto the Raspberry Pi: 
```  
scp <code_file> ece4723inter@ece4723inter.local:/home/ece4723inter/
```

### How to SSH into Raspberry Pi: 
```  
ssh ece4723inter@ece4723inter.local
```
password: `ece4723inter`

### How to SCP a file from Raspberry Pi to computer (for MAC): 
1. System Settings > General > Sharing
2. Find `Remote Login` and toggle it ON
``` 
scp <code_file> <username>@<local_hostname>:~/Desktop/
```
