-- Problem 01 (No. 175)
-- https://leetcode.com/problems/combine-two-tables/

SELECT FirstName, LastName, City, State FROM 
Person LEFT JOIN Address
USING (PersonId);


-- Problem 02 (No. 176)
-- https://leetcode.com/problems/second-highest-salary/

SELECT MAX(Salary) AS SecondHighestSalary FROM Employee
WHERE Salary < (SELECT MAX(Salary) FROM Employee);


-- Problem 03 (No. 177)
-- https://leetcode.com/problems/nth-highest-salary/

CREATE FUNCTION getNthHighestSalary(N IN NUMBER) RETURN NUMBER IS
result NUMBER;
BEGIN
    /* Write your PL/SQL query statement below */
    SELECT DISTINCT Salary INTO result FROM
        (SELECT Salary, DENSE_RANK() OVER (ORDER BY Salary DESC) Salary_Rank 
         FROM Employee
         ORDER BY Salary DESC)
    WHERE Salary_Rank = N;
    RETURN result;
END;


-- Problem 04 (No. 178)
-- https://leetcode.com/problems/rank-scores/

SELECT Score AS "score", DENSE_RANK() OVER (ORDER BY Score DESC) AS "Rank"
FROM Scores;


-- Problem 05 (No. 180)
-- https://leetcode.com/problems/consecutive-numbers/

SELECT DISTINCT l1.Num AS ConsecutiveNums FROM
Logs l1, Logs l2, Logs l3
WHERE l1.Num = l2.Num
AND l1.Num = l3.Num
AND l1.Id = l2.Id - 1
AND l1.Id = l3.Id - 2;
