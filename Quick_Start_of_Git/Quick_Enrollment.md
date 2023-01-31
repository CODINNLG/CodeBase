# Before Everything

## Easily Enroll
1. Accept my Invitation

2. Enter the downloaded repo & Config your own account
    - [Reference Link](https://blog.csdn.net/qq_33975041/article/details/104275499#:~:text=Linux%E4%B8%8B%E9%85%8D%E7%BD%AE%E6%9C%AC%E5%9C%B0git%E5%B9%B6%E5%90%8C%E6%AD%A5github%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%201%20%E5%AE%89%E8%A3%85git%E5%B7%A5%E5%85%B7%202%20%E5%AE%89%E8%A3%85SSH%E5%AF%86%E9%92%A5%203%20%E5%88%9B%E5%BB%BAgit,hub%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%204%20%E4%BD%BF%E7%94%A8git%20clone%E4%BB%8E%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E4%B8%8B%E8%BD%BD%E6%96%87%E4%BB%B6%205%20%E6%9C%AC%E5%9C%B0%E5%88%9B%E5%BB%BAhello_world.c%E6%96%87%E4%BB%B6%20%E5%B9%B6%E5%B0%86%E6%96%87%E4%BB%B6%E5%90%8C%E6%AD%A5%E5%88%B0%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93)
    - `git config --global user.email "you@example.com" `
    - ```git config --global user.name "Your Name"```
    - `git config --list`

3. Create your rsa_key and send to me
I add your key to the repo /  Add the rsa_key to your local house

4. How to access CODINNLG ?
- [reference link]( https://developer.aliyun.com/article/1081433)
- Follow through my personal access code:
Commond: `git clone https://Shared_Code_Base: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@github.com/CODINNLG/CodeBase`

5. Create and Check your own branch
Git 
- [reference link](https://www.freecodecamp.org/chinese/news/git-list-branches-how-to-show-all-remote-and-local-branch-names/)
- first change your own branch [must !!] 
- git branch -m main [your_branch_name]
- check your current branch with 
- commond: git branch -vv
- check all your branch: git branch -a
- check the existing remote branchs: git branch -r
- git branch -m [old_branch] [new_branch]
- git push origin [your_local_branch]
- **push your branch to the repo: one branch for each person! If you own more than 1 account, please delete the useless branch firstly and then update your new branch to the repo !**

6. Begin to push !
- [reference link1](https://blog.csdn.net/qq_33975041/article/details/104275499#:~:text=Linux%E4%B8%8B%E9%85%8D%E7%BD%AE%E6%9C%AC%E5%9C%B0git%E5%B9%B6%E5%90%8C%E6%AD%A5github%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%201%20%E5%AE%89%E8%A3%85git%E5%B7%A5%E5%85%B7%202%20%E5%AE%89%E8%A3%85SSH%E5%AF%86%E9%92%A5%203%20%E5%88%9B%E5%BB%BAgit,hub%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%204%20%E4%BD%BF%E7%94%A8git%20clone%E4%BB%8E%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E4%B8%8B%E8%BD%BD%E6%96%87%E4%BB%B6%205%20%E6%9C%AC%E5%9C%B0%E5%88%9B%E5%BB%BAhello_world.c%E6%96%87%E4%BB%B6%20%E5%B9%B6%E5%B0%86%E6%96%87%E4%BB%B6%E5%90%8C%E6%AD%A5%E5%88%B0%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93)
- [reference link2](https://blog.csdn.net/k663514387/article/details/110090886)
- git add [new_file]
- git status
- git commit -m "description"
- git push origin [local branch]:[remote branch] **Highly recommand !!**   or git push origin [remote branch]

# Common Errors
1. error: src refspec xxx does not match any / error: failed to push some refs to
- [reference](https://blog.csdn.net/u014361280/article/details/109703556)
2. `git switch` v.s. `git checkout`
- [reference](https://blog.csdn.net/iceboy314159/article/details/121375881) 
3. status is clean after git add
- [reference](https://blog.csdn.net/weixin_42433970/article/details/102796636)

# Advance Command
git branch
-v check all the states of your local branch
git checkout 
--[file path]  remove the ready added file from the local cache
update your git
git fetch origin --prune
