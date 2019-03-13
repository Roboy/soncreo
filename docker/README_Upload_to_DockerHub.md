# Getting an image to Docker Hub 
The Tutorial can also be found in [here](https://ropenscilabs.github.io/r-docker-tutorial/04-Dockerhub.html).
1. Imagine you made your own Docker image and would like to share it with the world you can sign up for an account on https://hub.docker.com/. After verifying your email you are ready to go and upload your first docker image.
2. Log in on https://hub.docker.com/ (if you do not have an account create one)
3. Click on Create Repository
4. Choose a name (e.g. soncreo) and a description for your repository and click Create.
5. Log into the Docker Hub from the command line 
    ```
    docker login --username=yourhubusername
    ```
    Enter your password when prompted. Then you should get the message `Login Succeeded`.
6. Save an existing docker image
    Check the image ID using: `docker images`

    and what you will see will be similar to
    
    ```
    REPOSITORY                    TAG                 IMAGE ID            CREATED             SIZE
    sharcc92/soncreo              latest              12cb16f7b0d3        15 minutes ago      88.1MB
7. Tag your image (save with an new name in the form `yourhubusername/respositoryname:newtag`.
    ```
    docker tag 12cb16f7b0d3  yourhubusername/soncreo:newtag
    ```
    The number must match the image ID and `:newtag` is the tag. In general, a good choice for a tag is something that will help you understand what this container should be used in conjunction with, or what it represents (e.g. `latest`, `v1`,`v2`)
8. Push your image to the repository you created
    ```
    docker push yourhubusername/respositoryname
9. Successful finished. Docker image can be publically downloaded by searching in [Dockerhub].

    
[Dockerhub]: https://hub.docker.com/

Further useful commands:
- Saving and loading images: `docker save soncreo > soncreo.tar`
- Loading images: `docker load --input verse_gapminder.tar`
