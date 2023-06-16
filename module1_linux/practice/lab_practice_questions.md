Q.1 Find the "root" string in /etc/passwd file and copy that string in /string.txt
Solution: `cat /etc/passwd | grep "root" > /string.txt`

Q.2 Find files which are owned by user "natasha" and copy them to "/data"
Solution: `find / -user natasha`

Q.3 Create a Command Shell alias on machine : 
- Create a command 'qstat'.
- It should able to execute the following command "ps -eo pid,tid,class,rtprio,ni,pri,psr,pcpu,stat,comm"
Solution: `alias qstat="ps -eo pid,tid,class,rtprio,ni,pri,psr,pcpu,stat,comm"`

Q. You need to change the owner of a file named /tmp/exe from harry, who is a member of the users group, to natasha, who is a member of the editors group. Assuming you want to change both user and group owners, which command will do this?
Solution: `chown natasha:editors /tmp/exe`

Q. You need to change the permissions of a file named schedule.txt such that the file owner can edit the file, users who are members of the group that owns the file can edit it, and users who are not owners and don't belong to the owning group can view it but not modify it. Which command will do this?
Solution: `chmod 774 schedule.txt`

Q. Display all the active connections on system using 'netstat' and grep source and destination IP and port number & for each port number print the associated file.

Q. Create script which create 50 users named as 'cdac_test_1' to 'cdac_test_50' with password as username. And create a file for alternate users starting from 1 in their home dir from root user name as 'username_flag.txt' and the data should be "username:data'.

```BASH

#!/bin/bash
#change start and end range in curly braces
for i in $(echo test{1..3})
do
    useradd -m $i
    var=`echo $i | tr -dc "[0-9]"`
    if [[ $(( var % 2 )) -ne 0 ]]
    then
        echo username:secret > /home/$i/usernameflag.txt
    fi
done

```

Q. Display all system info which includes:(/dev/sda)

a. CPU manufacturer (cat /proc/cpuinfo)
b. total number of cores available (cat /proc/cpuinfo)
c. Total number of memory in GB (free -mh)
d. Current free RAM in GB (free -mh)
e. total disk size present on system (lsblk)

Q. Script :::: Create a script /root/myscript.sh that should print "redhat" when user input "fedora" as argument, & vice-versa, it should print "redhat|fedora" without argument.
Solution: 

```BASH

#!/bin/bash
inp=$1
var1=redhat
var2=fedora
if [[ $inp == "$var1" ]]
then
    echo fedora
elif [[ $inp == "$var2" ]]
then
    echo redhat
elif [[ $# -le 0 ]]
then
    echo argument missing
else
    echo invalid input
fi

```

Q. How do you print the lines between 5 and 10, both inclusive?
Solution: `awk 'NR==5, NR==10' <filename>`

Q. Create a new file “new.txt” that is a concatenation of “file1.txt” and “file2.txt”?
Solution: `cat file1.txt file2.txt > new.txt`

Q. What is the output of the following code:

os=Unix
echo 1.$os 2."$os" 3.'$os' 4.$os
Solution: `1.os 2.os 3.os 4.os`

Q. If a file with execute permissions set, but with unknown file format is executed?
Solution: `linux does not care about the extension`

Q. To feed standard output of one command to standard input of another in a single shell session. example?
Solution: `ls > outputofls.txt`

Q. What is the output of this command?

cat < file1 >> file2 | file3

Solution: pipe takes the output of previous command and pipes it to next command.
file3 is not command
* `cat < file1` is same as `cat file1` 
* Operator >> appends the standard output to the file


Q. Syntax to suppress the display of command error to monitor?
Solution: `<command> 2> /dev/null`

Q. Find a .conf extension files in /etc and copy files to /search
Solution: `find /etc -type f -name "*.conf" 2> /dev/null -exec cp {} /search \;`

1. Create the following users, groups, and group memberships:
-A group named admin

-A user harry who belongs to admin as a secondary group

-A user natasha who also belongs to admin as a secondary group

-A user sarah who does not have access to an interactive shell on the system, and who is not a member of admin

-harry, natasha, and sarah should all have the password of "redhat@123?"

2. Create a collaborative directory /common/adm with the following characteristics:

-Group ownership of /common/adm is admin

-The directory should be readable, writable, and accessible to members of admin, but not to any other user. (It is understood that root has access to all ﬁles and directories on the system.)

Files created in /common/adm automatically have group ownership set to the admin group

3. The user natasha must conﬁgure a cron job that runs daily at 15:25 local time and executes
–/bin/echo hello 

Solution: `crontab -u natasha -e` to edit the cron file
`a b c d e /bin/echo hello`
`25 15 * * * /bin/echo hello`


4.Prepare a shell script (in BASH) named validatePassword.sh that receives one command line argument that is a string that will be checked whether it is valid to be used as a password. The necessary criteria that should be met in order to be a valid password are:
- It should contain at least 8 characters
- It should contain at least one uppercase letter and at least one lowercase letter.
Sample execution:
# ./validatePassword.sh weakpassword
INVALID

```BASH

!/bin/bash
inp=$1
no_of_charc=`echo $inp | wc -m`
upper_case=``
if [[ $# -le 0 ]]
then
    echo arguments missing
elif [[ $no_of_charc -lt 8 ]]
then
    echo weak password
    exit
elif [[ $inp =~ [A-Z] ]] && [[ $inp =~ [a-z] ]]
then
    echo valid
else
    echo invalid
fi

```

5.Consider the following output of bash terminal. Do not assume that any other commands are executed between the creation of this sample and the commands in the followed question. 
$ ls 
Assig Chapter 1 group 1 bak group2 assig1.c chapter chapter01 script1.sh slides student list files_list group1 chapter0[1-3] temp

What is the output of executing the following command line? 

$ Is *[!a-z][01] 
$ Is [A-Z]*[0-9] 

6.Write a bash script to create a menu driven program to do the following: 

- A welcome message with your name and UID
- Add menu:
	1. Add User 
	2. List number of Items 
	3. Check Permission 
	4. List Processes 
	5. Exit
NOTE: User should be able to choose any of the given options to perform the corresponding task.


7. Create a user mac with user id 3553
Solution: `useradd -u 3553 mac`

8. Find all files of user Jason and copy these files to /root/limit/.
Solution: `find -user Jason -exec cp {} /root/limit \;`

9.  Copy the ﬁle /etc/fstab to /var/tmp. Conﬁgure the permissions of /var/tmp/fstab so that: 

-The ﬁle /var/tmp/fstab is owned by the root user.

-The ﬁle /var/tmp/fstab belongs to the group root.

-The ﬁle /var/tmp/fstab should not be executable by anyone.

-The user harry is able to read and write /var/tmp/fstab.

-The user natasha can neither write nor read /var/tmp/fstab.

-all other users (current or future) have the ability to read /var/tmp/fstab 

10. Find a .conf extension files in /etc and copy files to /search

1. In /var/log/ display all the names of files which are having equal or more that 100 line.
Solution: `find -type f | xargs wc -l | head --lines=-1 | awk -F " " '$1>=100'`

2. create script which create 50 users named as 'cdac_test_1' ... 'cdac_test_50' with password as username. 
now create file for alternate users starting from 1 in their home dir as root user with filename as '$username_flag.txt' and the data should be "$username:SECRET'.

```BASH

#!/bin/bash
for i in $(echo test{1..3})
do
    useradd -m $i
    var=`echo $i | tr -dc "[0-9]"`
    if [[ $(( var % 2 )) -ne 0 ]]
    then
        echo username:secret > /home/$i/usernameflag.txt
    fi
done

```
