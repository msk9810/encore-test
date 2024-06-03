# mini3

### DB원격 접속 전 test data로 테이블 및 트리거 생성, csv파일 올리기

```sql
 CREATE TABLE users (
     user_id INT PRIMARY KEY AUTO_INCREMENT,
     name VARCHAR(20) UNIQUE,
     password VARCHAR(20) NOT NULL,
     best_score INT NOT NULL DEFAULT 0,
     average_score DOUBLE NOT NULL DEFAULT 0,
	   ranking INT,
     play_count INT NOT NULL DEFAULT 0
);

 CREATE TABLE games (
     game_id INT PRIMARY KEY AUTO_INCREMENT,
     user_id INT,
     FOREIGN KEY (user_id) REFERENCES users(user_id),
     when_played DATETIME NOT NULL,
     kill_count INT NOT NULL,
     elapsed_time DOUBLE NOT NULL,
     score INT NOT NULL
 );

DELIMITER //

CREATE TRIGGER update_user_scores
AFTER INSERT ON games
FOR EACH ROW
BEGIN
    -- Update bestScore
    UPDATE users u
    SET u.best_score = (
        SELECT MAX(g.score)
        FROM games g
        WHERE g.user_id = NEW.user_id
    )
    WHERE u.user_id = NEW.user_id;

    -- Update averageScore
    UPDATE users u
    SET u.average_score = (
        SELECT AVG(g.score)
        FROM games g
        WHERE g.user_id = NEW.user_id
    )
    WHERE u.user_id = NEW.user_id;

    -- Update playCount
    UPDATE users u
    SET u.play_count = (
        SELECT COUNT(*)
        FROM games g
        WHERE g.user_id = NEW.user_id
    )
    WHERE u.user_id = NEW.user_id;

    -- Update ranking
    UPDATE users u1
    JOIN (
        SELECT user_id, RANK() OVER (ORDER BY best_score DESC) as rnk
        FROM users
    ) ranking_data
    ON u1.user_id = ranking_data.user_id
    SET u1.ranking = ranking_data.rnk;
END //

DELIMITER ;

```

```python
##################################################################
# 주피터랩에서 csv파일들 mysql로 전송
# users 테이블 값 삽입
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# CSV 파일 경로
csv_file_path = 'C:\\Users\\Playdata\\Desktop\\Miniproject3\\user_data4.csv'

# CSV 파일 읽기
df_user = pd.read_csv(csv_file_path, encoding='utf-8')

# 데이터베이스 연결 엔진 생성
engine = create_engine('mysql+pymysql://root:1234@localhost:3306/mini1')

try:
    with engine.begin() as conn:  # 트랜잭션 시작
        # 데이터프레임의 각 행을 순회하면서 개별적으로 삽입
        for index, row in df_user.iterrows():
            name = str(row['name'])[:20]
            password = str(row['password'])[:20]
            
            # 중복된 name 값 확인
            result = conn.execute(text("SELECT COUNT(*) FROM users WHERE name = :name"), {"name": name})
            if result.scalar() > 0:
                print(f"Duplicate entry for name: {name}")
                continue  # 중복된 값이 있으면 삽입하지 않음
            
            # SQL 쿼리 생성
            sql = text("INSERT INTO users (name, password) VALUES (:name, :password)")
            
            # 개별 쿼리 실행
            conn.execute(sql, {"name": name, "password": password})
        
        print("Data has been successfully inserted into the users table.")
except Exception as e:
    print(f"An error occurred: {e}")

# games 테이블 삽입
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# CSV 파일 경로
csv_file_path = 'C:\\Users\\Playdata\\Desktop\\Miniproject3\\game_data4.csv'

# CSV 파일 읽기
df_game = pd.read_csv(csv_file_path, encoding='utf-8')

# 데이터베이스 연결 엔진 생성
engine = create_engine('mysql+pymysql://root:1234@localhost:3306/mini1')

try:
    with engine.begin() as conn:  # 트랜잭션 시작
        # 삽입 전 games 테이블 내용 확인
        result = conn.execute(text("SELECT * FROM games"))
        print("Before insertion:")
        for row in result:
            print(row)
        
        # NaN 값을 0으로 채워주기
        df_game.fillna(0, inplace=True)

        # 데이터프레임의 각 행을 순회하면서 개별적으로 삽입
        for index, row in df_game.iterrows():
            user_id = int(row['user_id'])
            when_played = datetime.strptime(row['when_played'], '%Y-%m-%d %H:%M')
            kill_count = int(row['kill_count'])
            elapsed_time = float(row['elapsed_time'])
            score = int(row['score'])
            
            # SQL 쿼리 생성
            sql = text("INSERT INTO games (user_id, when_played, kill_count, elapsed_time, score) VALUES (:user_id, :when_played, :kill_count, :elapsed_time, :score)")
            
            # 개별 쿼리 실행
            conn.execute(sql, {"user_id": user_id, "when_played": when_played, "kill_count": kill_count, "elapsed_time": elapsed_time, "score": score})
        
        # 삽입 후 games 테이블 내용 확인
        result = conn.execute(text("SELECT * FROM games"))
        print("After insertion:")
        for row in result:
            print(row)
        
    print("Data has been successfully inserted into the users table.")
except Exception as e:
    print(f"An error occurred: {e}")
```

![tesr data의 games 테이블](mini3%208f1ddc8735ae456f8f8e247739084fbf/245f1f67-f304-4849-a1a4-344ce38bdba2.png)

tesr data의 games 테이블

![test data 의 users 테이블](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled.png)

test data 의 users 테이블

[game_data4.csv](mini3%208f1ddc8735ae456f8f8e247739084fbf/game_data4.csv)

[user_data4.csv](mini3%208f1ddc8735ae456f8f8e247739084fbf/user_data4.csv)

# 외부 DB 연결

### 올릴 DB의 ip주소, 포트번호, username, 비밀번호를 치고 접속한다

![Untitled](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%201.png)

![Untitled](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%202.png)

접속완료

![Untitled](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%203.png)

아까 위에서 작성한 코드를 테스트 해보기 위해 test db하나 만든 후 jupyter lab으로 csv파일 삽입하려했는데 아래와 같은 에러가 뜬다.

![Untitled](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%204.png)

외부접속이라 보안때문에 추가적으로 패키지 설치를 해준다.

![Untitled](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%205.png)

```python
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# CSV 파일 경로
csv_file_path = 'C:\\Users\\Playdata\\Desktop\\Miniproject3\\user_data4.csv'

# CSV 파일 읽기
df_user = pd.read_csv(csv_file_path, encoding='utf-8')

# 데이터베이스 연결 엔진 생성
engine = create_engine('mysql+pymysql://client:7276@218.50.12.225:3306/test', connect_args={"ssl": {"ssl-ca": "/path/to/ca-cert.pem"}})

try:
    with engine.begin() as conn:  # 트랜잭션 시작
        # 데이터프레임의 각 행을 순회하면서 개별적으로 삽입
        for index, row in df_user.iterrows():
            name = str(row['name'])[:20]
            password = str(row['password'])[:20]
            
            # 중복된 name 값 확인
            result = conn.execute(text("SELECT COUNT(*) FROM users WHERE name = :name"), {"name": name})
            if result.scalar() > 0:
                print(f"Duplicate entry for name: {name}")
                continue  # 중복된 값이 있으면 삽입하지 않음
            
            # SQL 쿼리 생성
            sql = text("INSERT INTO users (name, password) VALUES (:name, :password)")
            
            # 개별 쿼리 실행
            conn.execute(sql, {"name": name, "password": password})
        
        print("Data has been successfully inserted into the users table.")
except Exception as e:
    print(f"An error occurred: {e}")
###############################################################################
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# CSV 파일 경로
csv_file_path = 'C:\\Users\\Playdata\\Desktop\\Miniproject3\\game_data4.csv'

# CSV 파일 읽기
df_game = pd.read_csv(csv_file_path, encoding='utf-8')

# 데이터베이스 연결 엔진 생성
engine = create_engine('mysql+pymysql://client:7276@218.50.12.225:3306/hehebalssa', connect_args={"ssl": {"ssl-ca": "/path/to/ca-cert.pem"}})

try:
    with engine.begin() as conn:  # 트랜잭션 시작
        # 삽입 전 games 테이블 내용 확인
        result = conn.execute(text("SELECT * FROM games"))
        print("Before insertion:")
        for row in result:
            print(row)
        
        # NaN 값을 0으로 채워주기
        df_game.fillna(0, inplace=True)

        # 데이터프레임의 각 행을 순회하면서 개별적으로 삽입
        for index, row in df_game.iterrows():
            user_id = int(row['user_id'])
            when_played = datetime.strptime(row['when_played'], '%Y-%m-%d %H:%M')
            kill_count = int(row['kill_count'])
            elapsed_time = float(row['elapsed_time'])
            score = int(row['score'])
            
            # SQL 쿼리 생성
            sql = text("INSERT INTO games (user_id, when_played, kill_count, elapsed_time, score) VALUES (:user_id, :when_played, :kill_count, :elapsed_time, :score)")
            
            # 개별 쿼리 실행
            conn.execute(sql, {"user_id": user_id, "when_played": when_played, "kill_count": kill_count, "elapsed_time": elapsed_time, "score": score})
        
        # 삽입 후 games 테이블 내용 확인
        result = conn.execute(text("SELECT * FROM games"))
        print("After insertion:")
        for row in result:
            print(row)
        
    print("Data has been successfully inserted into the users table.")
except Exception as e:
    print(f"An error occurred: {e}")
```

![원격으로 DB접속 후 test용 games table ](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%206.png)

원격으로 DB접속 후 test용 games table 

![원격으로 DB접속 후 test용 users table](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%207.png)

원격으로 DB접속 후 test용 users table

DB에 데이터들이 있어 따로 추가하지는 않고 트리거 추가 후 변화 확인 

![DB의 games table 데이터](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%208.png)

DB의 games table 데이터

![DB의 users table 데이터](mini3%208f1ddc8735ae456f8f8e247739084fbf/Untitled%209.png)

DB의 users table 데이터