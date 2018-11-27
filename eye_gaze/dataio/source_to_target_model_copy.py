"""
Script to change the namespace of a model
#Original Source: https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
"""

import sys, getopt
import tensorflow as tf
usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--checkpoint_save_dir=path/to/dir/ --replace_from=substr --replace_to=substr' \
            '--add_prefix=abc --dry_run'


def rename(checkpoint_dir, checkpoint_save_dir, replace_from, replace_to, add_prefix, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    # variables = tf.contrib.framework.list_variables(checkpoint_dir)

    reader = tf.contrib.framework.load_checkpoint(checkpoint_dir)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    variables = []
    for name in names:
        variables.append((name, variable_map[name]))

    # graph_target = tf.Graph()
    with tf.Session() as sess:
        for var_name, _ in variables:
            # Load the variable
            if var_name.endswith(":0"):
                var_name = var_name[:-2]
            var = reader.get_tensor(var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                # Rename the variable
                # with tf.Graph().as_default():
                if not 'Adam' in var_name:
                    print('Renaming %s to %s.' % (var_name, new_name))
                    var = tf.Variable(var, name=new_name)


        print("All variables renamed..")
        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            print("Saving model to ", checkpoint_save_dir)
            saver.save(sess, checkpoint_save_dir + 'model.ckpt')


def main(argv):
    checkpoint_dir = None
    checkpoint_save_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'checkpoint_save_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--checkpoint_save_dir':
            checkpoint_save_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    if not checkpoint_save_dir:
        print('Please specify a checkpoint_save_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, checkpoint_save_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])