import logging

logging.basicConfig(filename='app.log', level=logging.INFO,  format='%(asctime)s: module=%(module)10s: line=%(lineno)4s: message=%(message)s: function=%(funcName)s: process=%(process)d: level=%(levelname)s')
logger = logging.getLogger('main')


#
#