# Useful Console Commands

## Style

- prompt color change:
```console
echo $PS1
echo 'PS1="\$(date +%H:%M) \e[0;36m[\u@\h \w]$ \e[m "' >> ~/.bashrc 
```

- bash timestamp
```console
echo 'HISTTIMEFORMAT="%F %T "' >> ~/.bashrc
source ~/.bashrc
```

## Server Status
- storage
```console
df -h
du -h --max-depth=1 | sort -hr
```