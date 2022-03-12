# scp ./scp_all.sh ec2-user@18.217.154.51:~
# scp machinefile ec2-user@18.217.154.51:~
# scp init_sys.sh ec2-user@18.217.154.51:~
# scp init_silent.sh ec2-user@18.217.154.51:~
# scp ~/.ssh/id_rsa ec2-user@18.217.154.51:~/.ssh/id_rsa
# scp ./ssh_all.sh ec2-user@18.217.154.51:~
# scp ./ssh_parallel.sh ec2-user@18.217.154.51:~
# scp ./ssh_master.sh ec2-user@18.217.154.51:~

ssh ec2-user@18.217.154.51 './ssh_all.sh "echo hii"'
ssh ec2-user@18.217.154.51 './scp_all.sh .ssh/id_rsa'

# run init_silent on the master
# scp ~/.ssh/id_rsa ec2-user@18.217.154.51:~/.ssh/id_rsa
