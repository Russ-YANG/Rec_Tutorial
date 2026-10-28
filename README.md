# Rec_Tutorial

这是一个用基础写法实现推荐系统算法的个人练习项目。代码重点是展示算法步骤，
没有封装生产环境的数据校验、异常恢复和大规模检索优化。

## 环境

先运行 pip install -r requirements.txt 安装依赖。

主要示例使用转换后的 ml-100k/ratings.csv，字段为
userId,movieId,rating,timestamp。原始 MovieLens 100K 文件也保留在同一目录。

在项目根目录用模块方式运行示例，比如 python -m 召回.item_cf。

## 代码结构

common/ 放各个算法共用、和算法本身无关的代码：

- data.py：读数据、id 映射、划分训练集/测试集、用户-物品表、负采样、Dataset
- train.py：神经网络的训练循环，模型只需要提供 compute_loss(model, batch)
- layers.py：MLP（build_mlp）和权重初始化
- recommend.py：按分数排序、排除已交互物品、向量召回、排序模型逐个物品打分

每个算法文件开头的注释列出了算法步骤，每一步对应一个函数，可以单独阅读和测试。
召回、排序和 GNN 模型（ItemCF、DSSM、MMoE、DCN、LightGCN 等）的用法统一为：

    model = ItemCF(top_k=10, lastn=20).fit(load_ratings())
    model.recommend(user_id, n_rec=10)  # [(item_id, score), ...]

FM、SENet、Bilinear、FiBiNET、LHUC/PPNet、DIN、SIM 只实现网络结构本身，输入是已经准备好的特征。
SIM 复用了 DIN 的 attention 函数，Swing 继承自 ItemCF。

transformer/ 拆成了几个文件：

- model.py：注意力、多头、Encoder/Decoder、位置编码和手写的语言模型
- torch_version.py：用 nn.TransformerEncoder 实现的同一个模型
- data.py：分词器和英文句子数据集
- train.py：训练两个模型并对比，运行 python -m transformer.train

多样性的对比示例：python -m 多样性.dpp

## 练习场（网页）

在项目根目录运行（需要装了 torch 和 pytest 的环境）：

    python -m practice

浏览器会打开 http://127.0.0.1:8765。左边选一个函数，编辑器里是挖空的模板（保留函数签名和文档字符串），
写完点“提交测试”或按 ⌘/Ctrl + Enter。

- 提交时只替换这一个函数，在临时目录里运行这个模块的全部测试，不会改动项目里的文件
- 没通过的题会显示原因分析（数值不对、形状不匹配、键不存在等）、你的代码里出错的那一行、
  测试代码本身，以及完整报错
- 草稿和每道题的通过状态保存在浏览器里，可以随时看参考实现
- 服务器只监听本机，提交的代码会在本机运行

## 小测验

每个模块都有对应的测试 tests/test_<模块名>.py，题目用手算的小数据检查公式、
形状和关键性质（比如因果掩码、padding 不参与 attention）。

- python quiz.py：查看所有模块的完成情况
- python quiz.py item_cf：查看某个模块每道题的结果，加 -v 显示完整报错
- python quiz.py item_cf --watch：保存代码后自动重新测试

想重新练习某个模块时，把函数体换成 raise NotImplementedError，
总览里会显示为“还没开始”，写完后再跑一遍即可。也可以只挖空一个函数，
比如 召回/item_cf.py 里的 item_similarity，题目名会告诉你哪一步没写对。

## 原始数据说明
- **u.data**：主评分数据，结构为：
  
  | user_id | item_id | rating | timestamp |
  | ------- | ------- | ------ | --------- |
  
- **u.genre**：电影类型列表
- **u.info**：包含用户数、电影数和评分数等统计信息
- **u.item**：电影信息，结构为：
  
  | movie_id | movie_title | release_date | video_release_date | IMDb_URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
  | -------- | ----------- | ------------ | ------------------ | -------- | ------- | ------ | --------- | --------- | ---------- | ------ | ----- | ----------- | ----- | ------- | --------- | ------ | ------- | ------- | ------- | ------ | -------- | --- | ------- |
  
  后面各列为类型的独热编码

- **u.occupation**：职业列表
- **u.user**：用户信息，结构为：
  
  | user_id | age | gender | occupation | zip_code |
  | ------- | --- | ------ | ---------- | -------- |

- **u1.base/u1.test ... u5.base/u5.test**：5折交叉验证划分的数据子集
