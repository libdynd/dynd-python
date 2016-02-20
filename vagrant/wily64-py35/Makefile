start:
	vagrant up

kill: stop
	vagrant destroy --force

rebuild: | kill start

stop:
	vagrant halt

restart: | stop start

login: start
	vagrant ssh
