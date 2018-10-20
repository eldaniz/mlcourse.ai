set scp_cmd=C:\Proj\Utils\scp\WinSCP.exe
%scp_cmd% /log=winscp.log root:12345@10.0.75.2:5022 /command "get /root/.local/share/jupyter/runtime/*.json D:\DiskD\Eldaniz\ML\MLCourse\remote_kernels\"


