from tensortemplates.module import template_module
from wacacore.util.io import PassThroughOptionParser

def add_additional_options(argv):
    """Add options which only exist depending on other options"""
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1,
                      type='string')
    (poptions, args) = parser.parse_args(argv)
    # Get default options
    options = {}
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template

    # Get template specific options
    options.update(template_module[options['template']].kwargs())
    return options
