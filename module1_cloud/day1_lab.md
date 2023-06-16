# Day1 Lab Assignment

1. create a nginx container using nginx image and bind to port 8020 (here create your websire in html. copy the html file to host machine to container using docker volume mapping)
Solution:
`docker pull httpd:latest` && `docker pull nginx:latest`
`docker run -d --name "nginx-website" -p 8020:80 -v /home/darkstar/foo/nginx_website:/usr/share/nginx/html nginx:latest`

2. create a httpd container using nginx image and bind to port 8000 (here create your websire in html. copy the html file to host machine to container using docker volume mapping)
Solution:
`docker run -d --name "httpd-website1" -p 8000:80  -v /home/darkstar/foo/httpd_website1:/usr/local/apache2/htdocs httpd:latest`

3. create two container using same httpd image and bind to two different port 9000 and 9010 (here create your websire in html. copy the html file to host machine to container using docker volume mapping)
Solution:
`docker run -d --name "httpd-website2" -p 9000:80  -v /home/darkstar/foo/httpd_website2:/usr/local/apache2/htdocs httpd:latest`
`docker run -d --name "httpd-website3" -p 9010:80  -v /home/darkstar/foo/httpd_website3:/usr/local/apache2/htdocs httpd:latest`
