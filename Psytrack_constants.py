####################################### DATA PATHS ###########################

IN_DATA_PATH = '' # **<write here the path for the data>**
OUT_DATA_PATH = '' # **<write here the path for the output files>**
DATA_DICTS_PATH = ''  # **<write here the path for the output dataa dictionaries files>**

####################################### DATA PATHS ###########################

PICKL_EXTENSION = '.pkl'
AUDITORY_STIMULUS = 'auditory stimulus'
AUDITORY_STIMULUS_GO = 'Go stimulus'
AUDITORY_STIMULUS_NO_GO = 'NoGo stimulus'
GO_NOGO_ANALYSIS = "Go/NoGO Analysis"
AUDITORY_ANALYSIS = "Auditory Analysis"

PREV_STIM = 'previous stimulus'
PREV_ANSWER = 'previous answer'
PREV_ACTION = 'previous action'
MOUSE_NAME = "mouse_name"
LEVEL_NUM = "level"
TIME = "time"
LEVEL_NAME = 'levelname'
STIM_ID = "stimID"
SCORE = "score"
GO_WEIGHT = "Go stimulus"
LICK_BIAS = "Lick Bias"
NO_GO_WEIGHT = "NoGo stimulus"
HAB_LEVELS = ['Hab', 'Association', 'AssociationHab', 'Hab_ido',
              'Association_15_ido', 'Association_ido']
STIM_TO_REMOVE = [6, 7, 8]
NO_GO_LST = [2, 3, 4, 5]
NO_GO_STIM_ID_2 = 2
NO_GO_STIM_ID_3 = 3
NO_GO_STIM_ID_4 = 4
NO_GO_STIM_ID_5 = 5
NO_GO_STIM_ID_6 = 6
MIN_LICK_SCORE = 2

GO = 1

NO_GO = 0
NO_GO_REV = 0
NO_GO_06 = -0.6
NO_GO_08 = -0.8
NO_GO_09 = -0.9
LICK = 1
NO_LICK = 0

DELTA_MINUTES = 0
DELTA_HOURS = 2
MIN_SESSION_NUM = 10

RES_INDX = 0
PARAM_INDX = 1
RES_DICT = '{}-res_dict'
PARAM_DICT = '{}-param_dict'
PSY_DICT = '{}-psy_dict'

W_BIAS_IND = 1
W_AUDITORY_IND = 0
W_PREV_ACTION_IND = 2
AVG_BIN_SIZE = 20
FIRT_DATA_POINTS_PERCENTAGE = 0.2

ALL_OPT_TYPES = [['sigma', 'sigDay']]

COLORS = {'bias': '#FAA61A', AUDITORY_STIMULUS: "#A9373B",
          PREV_STIM: '#99CC66', PREV_ANSWER: '#9593D9', PREV_ACTION: '#59C3C3',
          GO_WEIGHT: "#A9373B", NO_GO_WEIGHT: "#2369BD", LICK_BIAS: '#FAA61A'}
ZORDER = {'bias': 2, AUDITORY_STIMULUS: 3, PREV_STIM: 3, PREV_ANSWER: 3,
          PREV_ACTION: 3, GO_WEIGHT: 3, NO_GO_WEIGHT: 3, LICK_BIAS: 2}
