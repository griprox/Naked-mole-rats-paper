from collections import defaultdict
# colors and orders

ORDER = {'colony' : ['baratheon', 'dothrakib','martell', 'stark', 'targaryen'],
         'sex' : ['M', 'F'],
         'rank' : ['rank1', 'rank2', 'rank3', 'rank4', 'rank5'], 
         'epoch' : ['Q#1', 'A#1', 'Q#2', 'A#2', '?',], 
         }

ORDER = defaultdict(lambda : None, ORDER)


COLOR = {'colony' : {'baratheon' : [0.97, 0.71, 0.23],
                     'targaryen' : [0.64, 0.17, 0.16],
                     'martell' : [0.91, 0.24, 0.13], 
                     'stark' : [0.24, 0.22, 0.22], 
                     'dothrakib' :[0.78,0.47,0.67],},
         
        'sex' : {'M' : [0.10, 0.10, 0.10],
                 'F' : [0.50, 0.50, 0.50]},
        'rank' : { 'rank1' : [0.91, 0.24, 0.13], 
                  'rank2' : [0.24, 0.22, 0.22], 
                  'rank3' :[0.00,0.76,0.76],
                  'rank4' : [0.42,0.78,0.92],
                  'rank5' :[0.65,0.21,0.55]}}
COLOR = defaultdict(lambda : None, COLOR)

EPOCHS = {'baratheon' : {('01-01-01', '03-04-20') : 'Q#1', 
                         ('04-04-20', '20-06-20') : 'A#1'},
          'stark' : {('01-01-17', '24-09-17') : 'Q#1',
                     ('25-09-17', '25-11-17') : 'A#1',
                     ('26-11-17', '01-02-18') : 'Q#2',
                     ('02-02-18', '30-06-18') : 'A#2'}
         }
EPOCHS = defaultdict(lambda : None, EPOCHS)

         