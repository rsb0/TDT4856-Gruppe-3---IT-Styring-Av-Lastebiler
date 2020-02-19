import React from 'react';
import {StyleShseet, Text, View, TouchableOpacity, ImageBackground, Dimensions, Button, StyleSheet} from 'react-native';
import { Camera } from 'expo-camera';
import { NavigationEvents } from "react-navigation";
import { Ionicons } from '@expo/vector-icons';
import Colors from "../constants/Colors";

export default class CameraComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            hasPermission: false,
            cameraType: Camera.Constants.Type.back,
            openCamera: true,
        };
        this.toggleCamera = this.toggleCamera.bind(this);
        this.retakeImage = this.retakeImage.bind(this);
        this.sendImage = this.sendImage.bind(this);
    }

    async componentDidMount() {
        const status = await Camera.requestPermissionsAsync();
        this.setState(state => ({
            hasPermission: status.granted
        }));
    }

    async takeImage() {
        if (this.camera && this.state.hasPermission) {
            let image = await this.camera.takePictureAsync({base64: true, exif: true});
            this.setState(state => ({
                image: image
            }));
            this.toggleCamera();
        }
    }

    toggleCamera() {
        this.setState(state => ({
            openCamera: true
        }))
    }

    retakeImage() {
        this.setState(state => ({
            image: null,
            openCamera: true,
        }))
    }

    async sendImage() {
        try {
            let formdata = new FormData();
            formdata.append("img", { uri: this.state.image.uri, type: "image/jpeg", name: "img.jpeg" });

            let res = await fetch("https://fuelprice-server.azurewebsites.net/upload/image", {
                method: "POST",
                headers: {
                    "Content-Type": "multipart/form-data"
                },
                body: formdata
            });
            res = await res.text();
            console.log(res);

            this.setState(state => ({
                image: null,
                openCamera: false,
            }));
            this.props.navigation.navigate("Prices", {imageTaken: true});
        } catch (e) {
            console.log(e);
        }
    }

    render() {
        if (this.state.image) {
            return(
                <View style={{ flex: 1 }}>
                    <ImageBackground
                        style={{
                            flex: 1,
                            justifyContent: "flex-end",
                            width: Dimensions.get('window').width,
                            height: Dimensions.get('window').height
                        }}
                        source={{uri: this.state.image.uri}}
                    >
                        <View style={{flex: 0.1, justifyContent: "space-around", flexDirection: "row"}}>
                            <Ionicons
                                name={"md-redo"}
                                size={50}
                                color={"white"}
                                onPress={this.retakeImage}
                            />
                            <Ionicons
                                name={"md-send"}
                                size={50}
                                color={"white"}
                                onPress={this.sendImage}
                            />
                        </View>
                    </ImageBackground>
                </View>
            )
        } else if(this.state.openCamera) {
            return(
                <View style={{ flex: 1 }}>
                    <NavigationEvents
                        onDidBlur={payload => this.setState(state => ({
                            openCamera: false,
                        }))}>
                    </NavigationEvents>
                    <Camera
                        style={{ flex: 1, justifyContent: "flex-end" }}
                        type={this.state.cameraType}
                        ref={ref => {this.camera = ref}}
                    >
                        <TouchableOpacity
                            style={{
                                flex: 0.1,
                                alignItems: 'center',
                                justifyContent: "center"
                            }}
                            onPress={() => {
                                this.takeImage();
                            }}>
                            <Ionicons
                                name={"md-radio-button-off"}
                                size={50}
                                color={"white"}
                            />
                        </TouchableOpacity>
                    </Camera>
                </View>
            )
        } else {
            return(
                <NavigationEvents
                    onWillFocus={payload => this.setState(state => ({
                        openCamera: true,
                    }))}>
                </NavigationEvents>
            )
        }
    }
}

const styles = StyleSheet.create({
    imageButton: {
        flex: 1
    },
});