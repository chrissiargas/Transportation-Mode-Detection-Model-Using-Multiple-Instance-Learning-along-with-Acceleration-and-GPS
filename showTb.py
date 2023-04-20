from tensorboard import program
user = 3
tracking_address = './logs_user' + str(user) + '/fullModelTb'  # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()



