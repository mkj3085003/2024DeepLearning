import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU是真的快！

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')#题目&内容分开呗
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')#除杂吧
                #出现下面的情况就不考虑了，都是有问题的“古诗”
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                #拼接：['G寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。碧林青旧竹，绿沼翠新苔。芝田初雁去，绮树巧莺来。E']
                content = start_token + content + end_token 
                poems.append(content)
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序    默认升序排列
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    #上面就是把词都给分开
    #['G', '寒', '随', '穷', '律', '变', '，', '春', '逐', '鸟', '声', '开', '。', '初', '风', '飘', '带', '柳', '，', '晚', '雪', '间', '花', '梅', '。', '碧', '林', '青', '旧', '竹', '，', '绿', '沼', '翠', '新', '苔', '。', '芝', '田', '初', '雁', '去', '，', '绮', '树', '巧', '莺', '来', '。', 'E']
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 根据词频降序排列
    #[('，', 4), ('。', 4), ('初', 2), ('G', 1), ('寒', 1), ('随', 1), ('穷', 1), ('律', 1), ('变', 1), ('春', 1), ('逐', 1), ('鸟', 1), ('声', 1), ('开', 1), ('风', 1), ('飘', 1), ('带', 1), ('柳', 1), ('晚', 1), ('雪', 1), ('间', 1), ('花', 1), ('梅', 1), ('碧', 1), ('林', 1), ('青', 1), ('旧', 1), ('竹', 1), ('绿', 1), ('沼', 1), ('翠', 1), ('新', 1), ('苔', 1), ('芝', 1), ('田', 1), ('雁', 1), ('去', 1), ('绮', 1), ('树', 1), ('巧', 1), ('莺', 1), ('来', 1), ('E', 1)]
    words, _ = zip(*count_pairs)#解压 分开成两个结构 将词与词频分开到不同的元素列表中
    
    #('，', '。', '初', 'G', '寒', '随', '穷', '律', '变', '春', '逐', '鸟', '声', '开', '风', '飘', '带', '柳', '晚', '雪', '间', '花', '梅', '碧', '林', '青', '旧', '竹', '绿', '沼', '翠', '新', '苔', '芝', '田', '雁', '去', '绮', '树', '巧', '莺', '来', 'E')
    #(4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)  
    words = words[:len(words)] + (' ',)#每个诗后面都加上,''
    #('，', ... '来', 'E', ' ')
    word_int_map = dict(zip(words, range(len(words))))#映射到字典里面 键值对，每个word有一个唯一的id
    #{'，': 0, '。': 1, '初': 2, 'G': 3, '寒': 4, '随': 5, '穷': 6, '律': 7, '变': 8, '春': 9, '逐': 10, '鸟': 11, '声': 12, '开': 13, '风': 14, '飘': 15, '带': 16, '柳': 17, '晚': 18, '雪': 19, '间': 20, '花': 21, '梅': 22, '碧': 23, '林': 24, '青': 25, '旧': 26, '竹': 27, '绿': 28, '沼': 29, '翠': 30, '新': 31, '苔': 32, '芝': 33, '田': 34, '雁': 35, '去': 36, '绮': 37, '树': 38, '巧': 39, '莺': 40, '来': 41, 'E': 42, ' ': 43}
    #根据词频吧，高的在前面，简单，后面的出现几率小，用复杂的，感觉类似于哈夫曼编码
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]#生成vector,将汉字古诗用数字id进行表示了
    #[[3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 1, 2, 14, 15, 16, 17, 0, 18, 19, 20, 21, 22, 1, 23, 24, 25, 26, 27, 0, 28, 29, 30, 31, 32, 1, 33, 34, 2, 35, 36, 0, 37, 38, 39, 40, 41, 1, 42]]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size #生成batch的number
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        看下面的例子：x第一个是6，y的第一个是2
        就是根据6，来预测下一个‘汉字’2
        y是x左移一位，最后一位重复【最后两位相同啊==> 9,9  5,5】
        仔细看看就知道了，上面的代码就是把y处理成这样了。
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
       
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)#随机种子，初始化，保证了每次结果都相同
    #得到word的embedding，vocab_length个word（+1应该是因为在前期数据处理时加了' '【个人感觉】），每个embedding的维度是100
    word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    #模型初始化啦，确定各个参数，做好预备工作
    rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    rnn_model.to(device)#GPU
    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    #优化呗，用的RMSprop，学习率0.01
    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()#初始化，需要label&predict
    # loss_fun = torch.nn.CrossEntropyLoss() ==> 相当于softmax + log + nllloss
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.
    #开始训练了 100个epoch
    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)#生成batch
        n_chunk = len(batches_inputs)#分成了几块，数据集/batch_size  ==> 有多少个batch
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)
            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))#变量化
                y = Variable(torch.from_numpy(y ))
                x, y = x.to(device), y.to(device)
                pre = rnn_model(x)#这里就进入model，得到预测结果
                loss += loss_fun(pre , y)#与真实label对比得loss
                if index == 0:
                    _, pre = torch.max(pre, dim=1)#输出预测概率最大的那一个word
                    print('prediction', pre.data.tolist()) # 输出预测结果（现在都是数字id形式的）
                    print('b_y       ', y.data.tolist())   # 输出label，真正的古诗，也是数字id形式
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE #计算平均损失吧
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist())
            optimizer.zero_grad()#梯度归零
            loss.backward()#反向传播
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)#对所有的梯度乘以一个clip_coef，缓解梯度爆炸问题（小于1）
            optimizer.step()#通过梯度下降执行一步参数更新

        if epoch % 5 ==0:
            torch.save(rnn_model.state_dict(), './poem_generator_rnn')#每五个epoch保存一次model
            print("finish  save model")


def to_word(predict, vocabs):  # 预测的结果转化成汉字（输出结果的时候才用，训练的时候根本不用【咱们又不用看，所以上面都是数字id的形式】）
    sample = np.argmax(predict)
    # print(sample)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    # print(vocabs[sample])
    return vocabs[sample]

def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    # print(shige)
    # poem_sentences = poem.split('。')
    print("".join(shige))
    # for s in poem_sentences:
    #     if s != '' and len(s) > 10:
    #         print(s + '。')

#这里就是根据模型训练的结果，开始验证，作诗了
def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=100, lstm_hidden_dim=128)

    rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))#根据之前保存的模型参数，生成诗歌

    # 指定开始的字
    rnn_model.to(device)#GPU呗
    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input)).to(device)#变量化
        output = rnn_model(input, is_test=True)#进入模型，得出output【现在这个是数字id形式的诗歌】
        word = to_word(output.data.tolist()[-1], vocabularies)#变成汉字了
        poem += word #一个word一个word的生成、预测
        # 50个汉字的诗吧
        if len(poem) > 50:
            return poem
    return poem
    
run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。

pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))
pretty_print_poem(gen_poem("星"))

