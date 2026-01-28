
class AlgResult:
    def __init__(self):
        self.algorithms = {} # alg name: alg
        self.AlgReward = {}
        self.AlgPicked = {}  # Records what article each algorithm picks
        self.Armpool = [] # 记录id
        self.Labelpool = []
        self.AlgRegret = {}
        self.AlgRewardRatio_vsRandom = {}
        self.BatchCumlateRegret = {}
        self.BatchCumlateReward = {}
        self.RandomChoiceRegret = []
        self.user_id =[]
        self.tim = []
        self.Fri2Sat = []
        self.traintime = 0
        self.epochcount = 0



