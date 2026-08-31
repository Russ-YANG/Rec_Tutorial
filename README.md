# Rec_Tutorial

这是一个用基础写法实现推荐系统算法的个人练习项目。代码重点是展示算法步骤，
没有封装生产环境的数据校验、异常恢复和大规模检索优化。

## 环境

先运行 pip install -r requirements.txt 安装依赖。

主要示例使用转换后的 ml-100k/ratings.csv，字段为
userId,movieId,rating,timestamp。原始 MovieLens 100K 文件也保留在同一目录。

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
