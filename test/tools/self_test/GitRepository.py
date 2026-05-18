import os
from git.repo import Repo
from git.repo.fun import is_git_dir


class GitRepository(object):
    """
    git仓库管理
    """

    def __init__(self, local_path, repo_url, branch='develop', is_submodule=False):
        self.local_path = local_path
        self.repo_url = repo_url
        self.repo = None
        self.initial(repo_url, branch, is_submodule)

    def initial(self, repo_url, branch, is_submodule):
        """
        初始化git仓库
        :param repo_url:
        :param branch:
        :return:
        """
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        git_local_path = os.path.join(self.local_path, '.git')
        print(git_local_path)
        if (not is_git_dir(git_local_path) and not is_submodule) or (not os.path.exists(git_local_path) and is_submodule):
            self.repo = Repo.clone_from(repo_url, to_path=self.local_path, branch=branch)
        else:
            self.repo = Repo(self.local_path)

    def pull(self):
        """
        从线上拉最新代码
        :return:
        """
        self.repo.remotes.origin.pull()

    def branches(self):
        """
        获取所有分支
        :return:
        """
        branches = self.repo.remote().refs
        return [item.remote_head for item in branches if item.remote_head not in ['HEAD', ]]

    def commits(self):
        """
        获取所有提交记录
        :return:
        """
        commit_log = self.repo.git.log('--pretty={"commit":"%h","author":"%an","summary":"%s","date":"%cd"}',
                                       max_count=50,
                                       date='format:%Y-%m-%d %H:%M')
        log_list = commit_log.split("\n")
        return [eval(item) for item in log_list]
    
    def latest_commit(self):
        """
        获取最新的提交记录
        :return:
        """
        commit_log = self.repo.git.show('-s')
        return commit_log

    def tags(self):
        """
        获取所有tag
        :return:
        """
        return [tag.name for tag in self.repo.tags]

    def change_to_branch(self, branch):
        """
        切换分值
        :param branch:
        :return:
        """
        # self.repo.git.checkout(branch)
        if branch in self.repo.branches:
            # 分支已经存在于本地，直接切换
            self.repo.git.checkout(branch)
        else:
            # 分支不存在于本地，检查分支是否存在于远端
            remote_branch_name = f'origin/{branch}'
            if branch in self.repo.remote().refs:
                # 远端分支存在，创建并切换到本地分支
                self.repo.create_head(branch, commit=f'origin/{branch}')
                self.repo.heads[branch].set_tracking_branch(self.repo.remote().refs[branch])
                self.repo.heads[branch].checkout()
            else:
                print(f'The branch {branch} does not exist locally or remotely.')

    def change_to_commit(self, branch, commit):
        """
        切换commit
        :param branch:
        :param commit:
        :return:
        """
        self.change_to_branch(branch=branch)
        self.repo.git.reset('--hard', commit)

    def change_to_tag(self, tag):
        """
        切换tag
        :param tag:
        :return:
        """
        self.repo.git.checkout(tag)
    
   

if __name__ == '__main__':
    repo = GitRepository(local_path = "/data/dnn/zhuangxin/tecoal/test/tecotest", repo_url = "ssh://zhuangxin@10.10.30.1:29418/tecotest")
    branch_list = repo.branches()
    print(branch_list)
    # repo.change_to_branch('dev')
    repo.pull()